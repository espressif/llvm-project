//===-- RISCVEsp32P4MemIntrin.cpp - ESP32-P4 Memory Intrinsics ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements optimization passes for ESP32-P4 memory operations.
// It transforms standard memcpy operations into optimized instruction sequences
// using specialized SIMD instructions available on the ESP32-P4 processor.
//
// The pass analyzes memory copy operations based on:
// - Source address alignment (16-byte, 8-byte, or unalign)
// - Destination address alignment (16-byte, 8-byte, or unalign)
// - Copy size (constant divisible by 16, constant divisible by 8,
//   other constants, or variable)
//
// For different combinations of these factors, it generates specialized code:
// - Small copies (<16 bytes): Uses optimized load/store instruction sequences
// - Medium copies: Utilizes SIMD vector load/store operations
// - Large copies: Implements block-based copy loops with SIMD instructions
//
// Key optimizations include:
// - Using 128-bit SIMD registers (q0-q7) for bulk transfers
// - Specialized patterns for handling alignment boundaries
// - Loop unrolling for common copy sizes
// - Special handling for different alignment combinations
// - Efficient tail handling for non-power-of-two sizes
//
// The pass creates helper functions for complex patterns to avoid code bloat
// and handles both constant-size and variable-size memory copies.
//
//===----------------------------------------------------------------------===//
#include "RISCVEsp32P4MemIntrin.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-esp32-p4-mem-intrin"

// Command line option to enable the RISCVEsp32P4MemIntrin pass
cl::opt<bool> llvm::EnableRISCVEsp32P4MemIntrin(
    "riscv-esp32-p4-mem-intrin", cl::init(false),
    cl::desc("Enable loop unrolling and Remainder specific loop"));

static cl::opt<unsigned> MemIntrinUnrollThresholdDefault(
    "riscv-esp32-p4-mem-intrin-unroll-threshold", cl::init(150), cl::Hidden,
    cl::desc("Maximum memcpy size (in bytes) to fully unroll instead of "
             "generating a loop."));

// Common method to check if function exists and create call
static bool helperCallTypesMatch(Function *F, Value *DstAddr, Value *SrcAddr,
                                 Value *Size) {
  FunctionType *FTy = F->getFunctionType();
  return FTy->getNumParams() == 3 &&
         FTy->getParamType(0) == DstAddr->getType() &&
         FTy->getParamType(1) == SrcAddr->getType() &&
         FTy->getParamType(2) == Size->getType();
}

bool RISCVEsp32P4MemIntrinBase::useExistingHelperFunction(
    MemCpyInst *M, IRBuilder<> &Builder, const std::string &FuncName,
    Value *DstAddr, Value *SrcAddr, Value *Size) {

  // Check if function exists in TheModule
  if (Function *ExistingFunc = TheModule->getFunction(FuncName)) {
    if (!helperCallTypesMatch(ExistingFunc, DstAddr, SrcAddr, Size))
      return false;
    // If function exists, create call directly
    Builder.CreateCall(ExistingFunc, {DstAddr, SrcAddr, Size});
    M->eraseFromParent();
    return true;
  }

  return false;
}

// Common method to check if function exists and create call
bool RISCVEsp32P4MemIntrinBase::useExistingHelperFunction(
    IRBuilder<> &Builder, const std::string &FuncName, Value *DstAddr,
    Value *SrcAddr, Value *Size) {

  // Check if function exists in TheModule
  if (Function *ExistingFunc = TheModule->getFunction(FuncName)) {
    if (!helperCallTypesMatch(ExistingFunc, DstAddr, SrcAddr, Size))
      return false;
    // If function exists, create call directly
    Builder.CreateCall(ExistingFunc, {DstAddr, SrcAddr, Size});
    return true;
  }

  return false;
}

// Create new helper function with inline control parameter
Function *RISCVEsp32P4MemIntrinBase::createMemCpyHelperFunction(
    IRBuilder<> &Builder, const std::string &FuncName, Value *DstAddr,
    Value *SrcAddr, bool isInline) {

  // Create new function type
  FunctionType *FuncTy = FunctionType::get(
      Builder.getVoidTy(), {Builder.getInt32Ty(), Builder.getInt32Ty()}, false);

  // Create new function
  Function *MCFunc = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                      FuncName, TheModule);
  auto ArgIt = MCFunc->arg_begin();
  ArgIt->setName("dst");
  (++ArgIt)->setName("src");

  // Create function call
  CallInst *Call = Builder.CreateCall(MCFunc, {DstAddr, SrcAddr});

  // For non-inline functions, set tail call
  if (!isInline) {
    Call->setTailCallKind(CallInst::TCK_Tail);
  }

  // Set function attributes
  MCFunc->addFnAttr(Attribute::NoUnwind);

  if (isInline) {
    MCFunc->addFnAttr(Attribute::AlwaysInline);
    MCFunc->addFnAttr(Attribute::InlineHint);
  } else {
    MCFunc->addFnAttr(Attribute::NoInline);
  }

  return MCFunc;
}

Function *RISCVEsp32P4MemIntrinBase::createMemCpyHelperFunctionGeneric(
    IRBuilder<> &Builder, const std::string &FuncName, Value *DstAddr,
    Value *SrcAddr, Value *Size, bool isInline, bool usePointers) {

  // Choose pointer type or int32 type based on usePointers parameter
  Type *ParamType;
  if (usePointers) {
    ParamType = Builder.getPtrTy();
  } else {
    ParamType = Builder.getInt32Ty();
  }

  // Create new function type
  FunctionType *FuncTy = FunctionType::get(
      Builder.getVoidTy(), {ParamType, ParamType, Builder.getInt32Ty()}, false);

  // ponytail: erase stale same-name helper when ptr/i32 signature migrates.
  if (Function *Existing = TheModule->getFunction(FuncName)) {
    if (Existing->getFunctionType() != FuncTy)
      Existing->eraseFromParent();
  }

  // Create new function
  Function *MCFunc = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                      FuncName, TheModule);
  // Name args at create — callers that also setName are redundant but OK.
  auto ArgIt = MCFunc->arg_begin();
  ArgIt->setName("dst");
  (++ArgIt)->setName("src");
  (++ArgIt)->setName("size");

  // Create function call
  CallInst *Call = Builder.CreateCall(MCFunc, {DstAddr, SrcAddr, Size});

  // For non-inline functions, set tail call
  if (!isInline) {
    Call->setTailCallKind(CallInst::TCK_Tail);
  }

  // Set function attributes
  MCFunc->addFnAttr(Attribute::NoUnwind);

  if (isInline) {
    MCFunc->addFnAttr(Attribute::AlwaysInline);
    MCFunc->addFnAttr(Attribute::InlineHint);
  } else {
    MCFunc->addFnAttr(Attribute::NoInline);
  }

  return MCFunc;
}

Function *RISCVEsp32P4MemIntrinBase::createMemCpyHelperFunction(
    IRBuilder<> &Builder, const std::string &FuncName, Value *DstAddr,
    Value *SrcAddr, Value *Size, bool isInline) {
  return createMemCpyHelperFunctionGeneric(Builder, FuncName, DstAddr, SrcAddr,
                                           Size, isInline, false);
}

Function *RISCVEsp32P4MemIntrinBase::createMemCpyHelperFunctionPtr(
    IRBuilder<> &Builder, const std::string &FuncName, Value *DstAddr,
    Value *SrcAddr, Value *Size, bool isInline) {
  return createMemCpyHelperFunctionGeneric(Builder, FuncName, DstAddr, SrcAddr,
                                           Size, isInline, true);
}

Function *RISCVEsp32P4MemIntrinBase::createMemCpyHelperFunctionPtrNoSize(
    IRBuilder<> &Builder, const std::string &FuncName, Value *Dst, Value *Src,
    bool isInline) {
  Type *PtrTy = Builder.getPtrTy();
  FunctionType *FuncTy =
      FunctionType::get(Builder.getVoidTy(), {PtrTy, PtrTy}, false);
  Function *MCFunc = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                      FuncName, TheModule);
  auto ArgIt = MCFunc->arg_begin();
  ArgIt->setName("dst");
  (++ArgIt)->setName("src");
  CallInst *Call = Builder.CreateCall(MCFunc, {Dst, Src});
  if (!isInline)
    Call->setTailCallKind(CallInst::TCK_Tail);
  MCFunc->addFnAttr(Attribute::NoUnwind);
  MCFunc->addFnAttr(isInline ? Attribute::InlineHint : Attribute::NoInline);
  return MCFunc;
}

void RISCVEsp32P4MemIntrinBase::createLoopBlocks(Function *F,
                                                 BasicBlock *&EntryBB,
                                                 BasicBlock *&ForBodyBB,
                                                 BasicBlock *&ForCleanupBB) {

  // Create basic blocks
  EntryBB = BasicBlock::Create(F->getContext(), "entry", F);
  ForBodyBB = BasicBlock::Create(F->getContext(), "for.body", F);
  ForCleanupBB = BasicBlock::Create(F->getContext(), "for.cond.cleanup", F);
}

// Set loop metadata
void RISCVEsp32P4MemIntrinBase::setLoopMetadata(Instruction *TermInst) {

  MDNode *LoopID = MDNode::get(TermInst->getContext(), {});
  MDNode *LoopMD = MDNode::get(
      TermInst->getContext(),
      {MDString::get(TermInst->getContext(), "llvm.loop.mustprogress")});
  MDNode *LoopMetadata = MDNode::get(TermInst->getContext(), {LoopID, LoopMD});

  cast<BranchInst>(TermInst)->setMetadata("llvm.loop", LoopMetadata);
}

// Add helper function to handle load/store instruction generation
std::pair<Value *, Value *> RISCVEsp32P4MemIntrin::generateLoadInstructions(
    IRBuilder<> &Builder, Value *SrcAddr, MemCpyType Type) {
  switch (Type) {
  case MemCpyType::Src16_Dst16_Const16:
  case MemCpyType::Src16_Dst16_Const8:
  case MemCpyType::Src16_Dst8_Const16:
  case MemCpyType::Src16_Dst8_Const8:
    return createEspVld128Ip(Builder, SrcAddr);
  case MemCpyType::Src8_Dst16_Const16:
  case MemCpyType::Src8_Dst16_Const8:
  case MemCpyType::Src8_Dst8_Const16: {
    // Load L64 first, then H64, combine into 128-bit vector
    auto [L64Data, L64Ptr] = createEspVldL64Ip(Builder, SrcAddr);
    auto [H64Data, H64Ptr] = createEspVldH64Ip(Builder, L64Ptr);

    // Combine L64 and H64 into 128-bit vector
    // L64 is lower 64 bits, H64 is upper 64 bits
    Value *CombinedVector = Builder.CreateShuffleVector(
        L64Data, H64Data,
        ArrayRef<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

    return {CombinedVector, H64Ptr};
  }
  case MemCpyType::Src8_Dst8_Const8:
    return createEspVldH64Ip(Builder, SrcAddr);
  default:
    // For unhandled cases, return null vector and original pointer
    return {nullptr, SrcAddr};
  }
}

// Add helper function to handle load/store instruction generation
Value *RISCVEsp32P4MemIntrin::generateStoreInstructions(IRBuilder<> &Builder,
                                                        Value *VectorData,
                                                        Value *DstAddr,
                                                        MemCpyType Type) {
  switch (Type) {
  case MemCpyType::Src16_Dst16_Const16:
  case MemCpyType::Src16_Dst16_Const8:
    // Store 128-bit vector directly
    return createEspVst128Ip(Builder, VectorData, DstAddr);
  case MemCpyType::Src8_Dst16_Const16:
  case MemCpyType::Src8_Dst16_Const8: {
    // For Src8_Dst16, VectorData is already 128-bit, split into L64 + H64
    // Extract lower 64 bits (L64) - indices 0-7 of v16i8
    Value *L64Data = Builder.CreateShuffleVector(
        VectorData, VectorData, // Use same vector for both operands
        ArrayRef<int>{0, 1, 2, 3, 4, 5, 6, 7});
    // Extract upper 64 bits (H64) - indices 8-15 of v16i8
    Value *H64Data = Builder.CreateShuffleVector(
        VectorData, VectorData, // Use same vector for both operands
        ArrayRef<int>{8, 9, 10, 11, 12, 13, 14, 15});

    // Store L64 first, then H64
    Value *AfterL64 = createEspVstL64Ip(Builder, L64Data, DstAddr);
    return createEspVstH64Ip(Builder, H64Data, AfterL64);
  }
  case MemCpyType::Src16_Dst8_Const16:
  case MemCpyType::Src16_Dst8_Const8:
  case MemCpyType::Src8_Dst8_Const16: {
    // For Src16_Dst8 / Src8_Dst8 with 128-bit block: store L64 then H64
    // (dst 8-byte aligned, so we must store both halves for full 16-byte copy)
    Value *L64Data = Builder.CreateShuffleVector(
        VectorData, VectorData, ArrayRef<int>{0, 1, 2, 3, 4, 5, 6, 7});
    Value *H64Data = Builder.CreateShuffleVector(
        VectorData, VectorData, ArrayRef<int>{8, 9, 10, 11, 12, 13, 14, 15});
    Value *AfterL64 = createEspVstL64Ip(Builder, L64Data, DstAddr);
    return createEspVstH64Ip(Builder, H64Data, AfterL64);
  }
  case MemCpyType::Src8_Dst8_Const8:
    // Store 64-bit vector directly (H64 part)
    return createEspVstH64Ip(Builder, VectorData, DstAddr);
  default:
    // Return the original address for unhandled cases
    return DstAddr;
  }
}

// Process a complete data block
void RISCVEsp32P4MemIntrin::processDataBlock(IRBuilder<> &Builder,
                                             Value *&SrcAddr, Value *&DstAddr,
                                             MemCpyType Type, int BlockSize) {
  // Use local variables to track the current address in the loop
  Value *CurrentSrc = SrcAddr;
  Value *CurrentDst = DstAddr;

  // Vector to store loaded data for each block
  SmallVector<Value *, 8> LoadedVectors;

  // Load loop: collect vector data and update source pointer
  for (int J = 0; J < BlockSize; J++) {
    auto [VectorData, UpdatedSrc] =
        generateLoadInstructions(Builder, CurrentSrc, Type);
    if (VectorData) {
      LoadedVectors.push_back(VectorData);
    }
    CurrentSrc = UpdatedSrc;
  }

  // Store loop: use stored vector data and update destination pointer
  for (int J = 0; J < BlockSize && J < LoadedVectors.size(); J++) {
    Value *VectorData = LoadedVectors[J];
    CurrentDst =
        generateStoreInstructions(Builder, VectorData, CurrentDst, Type);
  }

  // Update the original pointer variables passed by reference
  SrcAddr = CurrentSrc;
  DstAddr = CurrentDst;
}

std::pair<Value *, Value *>
RISCVEsp32P4MemIntrin::createEspVld128Ip(IRBuilder<> &Builder, Value *Src) {
  Type *i32Ty = Builder.getInt32Ty();

  // Get new intrinsic declaration with _m suffix
  Function *IntrinsicFunc = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vld_128_ip_m, {});
  // Create intrinsic call, returns {vector_data, updated_pointer}
  Value *Call = Builder.CreateCall(
      IntrinsicFunc, {Src, ConstantInt::get(i32Ty, 16)}, "vld128ip_m");

  // Extract vector data and updated pointer from struct return
  Value *VectorData = Builder.CreateExtractValue(Call, 0);
  Value *UpdatedPtr = Builder.CreateExtractValue(Call, 1);

  return {VectorData, UpdatedPtr};
}

// Legacy interface for backward compatibility during migration
Value *RISCVEsp32P4MemIntrin::createEspVld128Ip(IRBuilder<> &Builder,
                                                Value *Src, int Index) {
  assert(Index >= 0 && Index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc = Intrinsic::getDeclarationIfExists(
      TheModule, Intrinsic::riscv_esp_vld_128_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {Src, ConstantInt::get(i32Ty, 16), ConstantInt::get(i32Ty, Index)},
      "vld128ip");
}

// Migrated to use intrinsic esp.vld.h.64.ip_m
// Returns {vector_data, updated_pointer}
std::pair<Value *, Value *>
RISCVEsp32P4MemIntrin::createEspVldH64Ip(IRBuilder<> &Builder, Value *Src) {
  Type *i32Ty = Builder.getInt32Ty();

  Function *IntrinsicFunc = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vld_h_64_ip_m, {});
  Value *Call = Builder.CreateCall(
      IntrinsicFunc, {Src, ConstantInt::get(i32Ty, 8)}, "vldh64ip_m");

  Value *VectorData = Builder.CreateExtractValue(Call, 0);
  Value *UpdatedPtr = Builder.CreateExtractValue(Call, 1);

  return {VectorData, UpdatedPtr};
}

// Legacy interface for backward compatibility during migration
Value *RISCVEsp32P4MemIntrin::createEspVldH64Ip(IRBuilder<> &Builder,
                                                Value *Src, int Index) {
  assert(Index >= 0 && Index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc = Intrinsic::getDeclarationIfExists(
      TheModule, Intrinsic::riscv_esp_vld_h_64_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {Src, ConstantInt::get(i32Ty, 8), ConstantInt::get(i32Ty, Index)},
      "vldh64ip");
}

// Migrated to use intrinsic esp.vld.l.64.ip_m
// Returns {vector_data, updated_pointer}
std::pair<Value *, Value *>
RISCVEsp32P4MemIntrin::createEspVldL64Ip(IRBuilder<> &Builder, Value *Src) {
  Type *i32Ty = Builder.getInt32Ty();

  Function *IntrinsicFunc = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vld_l_64_ip_m, {});
  Value *Call = Builder.CreateCall(
      IntrinsicFunc, {Src, ConstantInt::get(i32Ty, 8)}, "vldl64ip_m");

  Value *VectorData = Builder.CreateExtractValue(Call, 0);
  Value *UpdatedPtr = Builder.CreateExtractValue(Call, 1);

  return {VectorData, UpdatedPtr};
}

// Legacy interface for backward compatibility during migration
Value *RISCVEsp32P4MemIntrin::createEspVldL64Ip(IRBuilder<> &Builder,
                                                Value *Src, int Index) {
  assert(Index >= 0 && Index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc = Intrinsic::getDeclarationIfExists(
      TheModule, Intrinsic::riscv_esp_vld_l_64_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {Src, ConstantInt::get(i32Ty, 8), ConstantInt::get(i32Ty, Index)},
      "vldl64ip");
}

// Rename and modify: use intrinsic esp.vst.128.ip
// Return the updated dst pointer (i32)
Value *RISCVEsp32P4MemIntrin::createEspVst128Ip(IRBuilder<> &Builder,
                                                Value *VectorData, Value *dst) {
  Type *i32Ty = Builder.getInt32Ty();

  Function *IntrinsicFunc = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vst_128_ip_m, {});
  Value *CallResult = Builder.CreateCall(
      IntrinsicFunc, {VectorData, dst, ConstantInt::get(i32Ty, 16)},
      "vst128ip_m");

  return CallResult;
}

// Legacy interface for backward compatibility during migration
Value *RISCVEsp32P4MemIntrin::createEspVst128Ip(IRBuilder<> &Builder,
                                                Value *Dst, int Index) {
  assert(Index >= 0 && Index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc = Intrinsic::getDeclarationIfExists(
      TheModule, Intrinsic::riscv_esp_vst_128_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {ConstantInt::get(i32Ty, Index), Dst, ConstantInt::get(i32Ty, 16)},
      "vst128ip");
}

// Migrated to use intrinsic esp.vst.h.64.ip_m
// Returns updated dst pointer
Value *RISCVEsp32P4MemIntrin::createEspVstH64Ip(IRBuilder<> &Builder,
                                                Value *VectorData, Value *dst) {
  Type *i32Ty = Builder.getInt32Ty();

  Function *IntrinsicFunc = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vst_h_64_ip_m, {});
  Value *CallResult = Builder.CreateCall(
      IntrinsicFunc, {VectorData, dst, ConstantInt::get(i32Ty, 8)},
      "vsth64ip_m");

  return CallResult;
}

// Legacy interface for backward compatibility during migration
Value *RISCVEsp32P4MemIntrin::createEspVstH64Ip(IRBuilder<> &Builder,
                                                Value *Dst, int Index) {
  assert(Index >= 0 && Index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc = Intrinsic::getDeclarationIfExists(
      TheModule, Intrinsic::riscv_esp_vst_h_64_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {ConstantInt::get(i32Ty, Index), Dst, ConstantInt::get(i32Ty, 8)},
      "vsth64ip");
}

// Migrated to use intrinsic esp.vst.l.64.ip_m
// Returns updated dst pointer
Value *RISCVEsp32P4MemIntrin::createEspVstL64Ip(IRBuilder<> &Builder,
                                                Value *VectorData, Value *dst) {
  Type *i32Ty = Builder.getInt32Ty();

  Function *IntrinsicFunc = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_vst_l_64_ip_m, {});
  Value *CallResult = Builder.CreateCall(
      IntrinsicFunc, {VectorData, dst, ConstantInt::get(i32Ty, 8)},
      "vstl64ip_m");

  return CallResult;
}

// Legacy interface for backward compatibility during migration
Value *RISCVEsp32P4MemIntrin::createEspVstL64Ip(IRBuilder<> &Builder,
                                                Value *Dst, int Index) {
  assert(Index >= 0 && Index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc = Intrinsic::getDeclarationIfExists(
      TheModule, Intrinsic::riscv_esp_vst_l_64_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {ConstantInt::get(i32Ty, Index), Dst, ConstantInt::get(i32Ty, 8)},
      "vstl64ip");
}

enum MemCpyType RISCVEsp32P4MemIntrinBase::getMemCpyType(MemCpyInst *M) {
  MaybeAlign SrcAlign = M->getSourceAlign();
  SrcAlignValue = SrcAlign->value();
  // Determine the source alignment category
  SrcAlignment SrcAlignCat = SrcAlignment::SrcUnalign;
  if (isDivisibleBy16(SrcAlignValue))
    SrcAlignCat = SrcAlignment::Src16;
  else if (isDivisibleBy8(SrcAlignValue))
    SrcAlignCat = SrcAlignment::Src8;

  MaybeAlign DstAlign = M->getDestAlign();
  DstAlignValue = DstAlign->value();
  // Determine the destination alignment category
  DstAlignment DstAlignCat = DstAlignment::DstUnalign;
  if (isDivisibleBy16(DstAlignValue))
    DstAlignCat = DstAlignment::Dst16;
  else if (isDivisibleBy8(DstAlignValue))
    DstAlignCat = DstAlignment::Dst8;

  // Determine the length type
  SizeType SizeType = SizeType::Var;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(M->getLength())) {
    Len = CI->getZExtValue();
    if (isDivisibleBy16(Len))
      SizeType = SizeType::Const16;
    else if (isDivisibleBy8(Len))
      SizeType = SizeType::Const8;
    else
      SizeType = SizeType::OtherConst;
  }
  // for size variable, can't inline the function
  else {
    SizeValue = M->getLength();
  }

  // Three-dimensional conditional judgment
  switch (SrcAlignCat) {
  case SrcAlignment::Src16:
    switch (DstAlignCat) {
    case DstAlignment::Dst16:
      switch (SizeType) {
      case SizeType::Const16:
        return MemCpyType::Src16_Dst16_Const16;
      case SizeType::Const8:
        return MemCpyType::Src16_Dst16_Const8;
      case SizeType::OtherConst:
        return MemCpyType::Src16_Dst16_OtherConst;
      default:
        return MemCpyType::Src16_Dst16_Var;
      }
    case DstAlignment::Dst8:
      switch (SizeType) {
      case SizeType::Const16:
        return MemCpyType::Src16_Dst8_Const16;
      case SizeType::Const8:
        return MemCpyType::Src16_Dst8_Const8;
      case SizeType::OtherConst:
        return MemCpyType::Src16_Dst8_OtherConst;
      default:
        return MemCpyType::Src16_Dst8_Var;
      }
    default: // DstUnalign
      switch (SizeType) {
      case SizeType::Const16:
        return MemCpyType::Src16_DstUnalign_Const16;
      case SizeType::Const8:
        return MemCpyType::Src16_DstUnalign_Const8;
      case SizeType::OtherConst:
        return MemCpyType::Src16_DstUnalign_OtherConst;
      default:
        return MemCpyType::Src16_DstUnalign_Var;
      }
    }
  case SrcAlignment::Src8:
    switch (DstAlignCat) {
    case DstAlignment::Dst16:
      switch (SizeType) {
      case SizeType::Const16:
        return MemCpyType::Src8_Dst16_Const16;
      case SizeType::Const8:
        return MemCpyType::Src8_Dst16_Const8;
      case SizeType::OtherConst:
        return MemCpyType::Src8_Dst16_OtherConst;
      default:
        return MemCpyType::Src8_Dst16_Var;
      }
    case DstAlignment::Dst8:
      switch (SizeType) {
      case SizeType::Const16:
        return MemCpyType::Src8_Dst8_Const16;
      case SizeType::Const8:
        return MemCpyType::Src8_Dst8_Const8;
      case SizeType::OtherConst:
        return MemCpyType::Src8_Dst8_OtherConst;
      default:
        return MemCpyType::Src8_Dst8_Var;
      }
    default: // DstUnalign
      switch (SizeType) {
      case SizeType::Const16:
        return MemCpyType::Src8_DstUnalign_Const16;
      case SizeType::Const8:
        return MemCpyType::Src8_DstUnalign_Const8;
      case SizeType::OtherConst:
        return MemCpyType::Src8_DstUnalign_OtherConst;
      default:
        return MemCpyType::Src8_DstUnalign_Var;
      }
    }
  default: // SrcUnalign
    switch (DstAlignCat) {
    case DstAlignment::Dst16:
      switch (SizeType) {
      case SizeType::Const16:
        return MemCpyType::SrcUnalign_Dst16_Const16;
      case SizeType::Const8:
        return MemCpyType::SrcUnalign_Dst16_Const8;
      case SizeType::OtherConst:
        return MemCpyType::SrcUnalign_Dst16_OtherConst;
      default:
        return MemCpyType::SrcUnalign_Dst16_Var;
      }
    case DstAlignment::Dst8:
      switch (SizeType) {
      case SizeType::Const16:
        return MemCpyType::SrcUnalign_Dst8_Const16;
      case SizeType::Const8:
        return MemCpyType::SrcUnalign_Dst8_Const8;
      case SizeType::OtherConst:
        return MemCpyType::SrcUnalign_Dst8_OtherConst;
      default:
        return MemCpyType::SrcUnalign_Dst8_Var;
      }
    default: // DstUnalign
      switch (SizeType) {
      case SizeType::Const16:
        return MemCpyType::SrcUnalign_DstUnalign_Const16;
      case SizeType::Const8:
        return MemCpyType::SrcUnalign_DstUnalign_Const8;
      case SizeType::OtherConst:
        return MemCpyType::SrcUnalign_DstUnalign_OtherConst;
      default:
        return MemCpyType::SrcUnalign_DstUnalign_Var;
      }
    }
  }
}

// Generic memory copy processing function
bool RISCVEsp32P4MemIntrinPass::processMemCpyWithAlignment(
    MemCpyType Type, MemCpyInst *M, BasicBlock::iterator &BBI,
    const std::string &FuncName, uint64_t BlockSize, uint64_t ChunkSize) {

  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  uint64_t Times = Len / ChunkSize;
  Value *SrcAddr = nullptr;
  Value *DstAddr = nullptr;

  // Len exceeds the specified size, need for loop
  if (Times > 8) {
    uint64_t totalBlocks = Len / BlockSize;
    uint64_t Remainder = Len % BlockSize;
    Times = Remainder / ChunkSize;
    SrcAddr = Src;
    DstAddr = Dst;

    // When totalBlocks loop count exceeds threshold, do not expand using loop
    if (totalBlocks > MemIntrinUnrollThresholdDefault) {
      // First check if function exists in current TheModule
      if (useExistingHelperFunction(M, Builder, FuncName, DstAddr, SrcAddr,
                                    Builder.getInt32(Len))) {
        return true;
      }

      // Create loop processing function with ptr args, must not inline
      Function *MCFunc = createMemCpyHelperFunctionPtr(
          Builder, FuncName, DstAddr, SrcAddr, Builder.getInt32(Len), false);

      BasicBlock *EntryBB = nullptr, *ForBodyBB = nullptr,
                 *ForCleanupBB = nullptr;
      createLoopBlocks(MCFunc, EntryBB, ForBodyBB, ForCleanupBB);

      IRBuilder<> FuncBuilder(EntryBB);
      Function::arg_iterator ArgIt = MCFunc->arg_begin();
      Value *Dst = ArgIt++;
      Value *Src = ArgIt++;
      Value *Size = ArgIt++;

      Value *Div = FuncBuilder.CreateLShr(Size, FuncBuilder.getInt32(7));
      Value *Cmp =
          FuncBuilder.CreateICmpULT(Size, FuncBuilder.getInt32(BlockSize));
      FuncBuilder.CreateCondBr(Cmp, ForCleanupBB, ForBodyBB);

      FuncBuilder.SetInsertPoint(ForBodyBB);
      PHINode *I = FuncBuilder.CreatePHI(Builder.getInt32Ty(), 2);
      I->addIncoming(FuncBuilder.getInt32(0), EntryBB);

      // Create PHI nodes for source and destination addresses (ptr), used to
      // track the current address being processed in the loop
      PHINode *SrcPtrLoop =
          FuncBuilder.CreatePHI(Builder.getPtrTy(), 2, "src.ptr.loop");
      SrcPtrLoop->addIncoming(Src, EntryBB);
      Value *SrcPtrInit = SrcPtrLoop;
      PHINode *DstPtrLoop =
          FuncBuilder.CreatePHI(Builder.getPtrTy(), 2, "dst.ptr.loop");
      DstPtrLoop->addIncoming(Dst, EntryBB);
      Value *DstPtrInit = DstPtrLoop;

      // Generate instructions based on different load/store styles
      processDataBlock(FuncBuilder, SrcPtrInit, DstPtrInit, Type, 8);
      SrcPtrLoop->addIncoming(SrcPtrInit, ForBodyBB);
      DstPtrLoop->addIncoming(DstPtrInit, ForBodyBB);

      Value *Inc =
          FuncBuilder.CreateAdd(I, FuncBuilder.getInt32(1), "", true, true);
      Value *ExitCond = FuncBuilder.CreateICmpEQ(Inc, Div);
      FuncBuilder.CreateCondBr(ExitCond, ForCleanupBB, ForBodyBB);
      I->addIncoming(Inc, ForBodyBB);

      FuncBuilder.SetInsertPoint(ForCleanupBB);
      // Create PHI nodes for source and destination addresses (ptr)
      PHINode *SrcPtrCleanup =
          FuncBuilder.CreatePHI(Builder.getPtrTy(), 2, "src.ptr.cleanup");
      SrcPtrCleanup->addIncoming(Src, EntryBB);
      SrcPtrCleanup->addIncoming(SrcPtrInit, ForBodyBB);
      Value *SrcPtrCleanupInit = SrcPtrCleanup;
      PHINode *DstPtrCleanup =
          FuncBuilder.CreatePHI(Builder.getPtrTy(), 2, "dst.ptr.cleanup");
      DstPtrCleanup->addIncoming(Dst, EntryBB);
      DstPtrCleanup->addIncoming(DstPtrInit, ForBodyBB);
      Value *DstPtrCleanupInit = DstPtrCleanup;
      // The remaining Remainder part is directly generated
      processDataBlock(FuncBuilder, SrcPtrCleanupInit, DstPtrCleanupInit, Type,
                       Times);
      FuncBuilder.CreateRetVoid();
      setLoopMetadata(ForBodyBB->getTerminator());

    } else {
      // Fully expand
      for (uint64_t I = 0; I < totalBlocks; I++) {
        processDataBlock(Builder, SrcAddr, DstAddr, Type, 8);
      }

      processDataBlock(Builder, SrcAddr, DstAddr, Type, Times);
    }

  } else {
    // Len does not exceed the specified size, can be processed in one go
    SrcAddr = Src;
    DstAddr = Dst;
    // Directly expand to handle small data (ptr passed directly)
    processDataBlock(Builder, SrcAddr, DstAddr, Type, Times);
  }

  // Process possible additional remaining parts (e.g. src16dst8const8 and
  // src8dst16const8 at the end of the function) This part needs to be added
  // based on actual conditions
  switch (Type) {
  case MemCpyType::Src16_Dst16_Const8:
  case MemCpyType::Src8_Dst16_Const8:
  case MemCpyType::Src16_Dst8_Const8: {
    auto [VectorData, UpdatedSrc] = createEspVldL64Ip(Builder, SrcAddr);
    DstAddr = createEspVstL64Ip(Builder, VectorData, DstAddr);
    SrcAddr = UpdatedSrc;
    break;
  }
  default:
    break;
  }
  M->eraseFromParent();
  return true;
}

// src 16-byte aligned, dst 16-byte aligned, size divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrc16Dst16Const16(
    MemCpyType Type, MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processMemCpyWithAlignment(Type, M, BBI,
                                    "esp32p4MemCpySrc16Dst16Const16", 128, 16);
}

// src 16-byte aligned, dst 16-byte aligned, size divisible by 8
bool RISCVEsp32P4MemIntrinPass::processSrc16Dst16Const8(
    MemCpyType Type, MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processMemCpyWithAlignment(Type, M, BBI,
                                    "esp32p4MemCpySrc16Dst16Const8", 128, 16);
}

// src 16-byte aligned, dst 8-byte aligned, size divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrc16Dst8Const16(
    MemCpyType Type, MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processMemCpyWithAlignment(Type, M, BBI,
                                    "esp32p4MemCpySrc16Dst8Const16", 128, 16);
}

// src is 16-byte aligned, dst is 8-byte aligned, size is divisible by 8
bool RISCVEsp32P4MemIntrinPass::processSrc16Dst8Const8(
    MemCpyType Type, MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processMemCpyWithAlignment(Type, M, BBI, "esp32p4MemCpySrc16Dst8Var",
                                    128, 16);
}

// src 8-byte aligned, dst 16-byte aligned, size divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrc8Dst16Const16(
    MemCpyType Type, MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processMemCpyWithAlignment(Type, M, BBI,
                                    "esp32p4MemCpySrc8Dst16Const16", 128, 16);
}

// src 8-byte aligned, dst 16-byte aligned, size divisible by 8
bool RISCVEsp32P4MemIntrinPass::processSrc8Dst16Const8(
    MemCpyType Type, MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processMemCpyWithAlignment(Type, M, BBI,
                                    "esp32p4MemCpySrc8Dst16Const8", 128, 16);
}

// src 8-byte aligned, dst 8-byte aligned, size divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrc8Dst8Const16(
    MemCpyType Type, MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processMemCpyWithAlignment(Type, M, BBI,
                                    "esp32p4MemCpySrc8Dst8Const16", 128, 16);
}

// src 8-byte aligned, dst 8-byte aligned, size divisible by 8
bool RISCVEsp32P4MemIntrinPass::processSrc8Dst8Const8(
    MemCpyType Type, MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processMemCpyWithAlignment(Type, M, BBI, "esp32p4MemCpySrc8Dst8Const8",
                                    64, 8);
}

// it supports 1-15 bytes
// src 16| 8 align and dst 16| 8 align
bool RISCVEsp32P4MemIntrinPass::processSrc16Dst16From1To15Const(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  IRBuilder<> Builder(M);
  Value *OrigSrc = M->getSource(); // Keep original pointers
  Value *OrigDst = M->getDest();
  Value *CurrentSrc = OrigSrc; // Pointers to use for copying
  Value *CurrentDst = OrigDst;

  ConstantInt *LenCI = dyn_cast<ConstantInt>(M->getLength());
  if (!LenCI)
    return false;
  uint64_t Len = LenCI->getZExtValue();

  assert(Len > 0 && Len < 16 && "Len must be between 1 and 15");

  Type *I8Ty = Builder.getInt8Ty();
  Type *I16Ty = Builder.getInt16Ty();
  Type *I32Ty = Builder.getInt32Ty();

  uint64_t BytesCopied = 0;

  // If length >= 8, prioritize using 8-byte copy (ptr passed directly)
  if (Len >= 8) {
    auto [VectorData, UpdatedSrc] = createEspVldL64Ip(Builder, CurrentSrc);
    CurrentDst = createEspVstL64Ip(Builder, VectorData, CurrentDst);
    CurrentSrc = UpdatedSrc;
    BytesCopied = 8;
  }

  // --- Use LLVM IR to handle remaining bytes (Len - BytesCopied) ---

  // Handle remaining 4 bytes
  if (Len - BytesCopied >= 4) {
    handleRemainingBytes(Builder, I32Ty, I8Ty, CurrentSrc, CurrentDst, 4);
    BytesCopied += 4;
  }

  // Handle remaining 2 bytes
  if (Len - BytesCopied >= 2) {
    handleRemainingBytes(Builder, I16Ty, I8Ty, CurrentSrc, CurrentDst, 2);
    BytesCopied += 2;
  }

  // Handle remaining 1 byte
  if (Len - BytesCopied >= 1) {
    Value *LoadVal = Builder.CreateAlignedLoad(I8Ty, CurrentSrc, Align(1));
    Builder.CreateAlignedStore(LoadVal, CurrentDst, Align(1));
    BytesCopied += 1;
  }

  // Remove the original memcpy instruction
  M->eraseFromParent();
  return true;
}

void RISCVEsp32P4MemIntrinPass::handleRemainingBytes(
    IRBuilder<> &Builder, Type *I16TimesTy, Type *I8Ty, Value *&CurrentSrc,
    Value *&CurrentDst, int BytesNum) {
  Value *LoadVal =
      Builder.CreateAlignedLoad(I16TimesTy, CurrentSrc, Align(BytesNum));
  Builder.CreateAlignedStore(LoadVal, CurrentDst, Align(BytesNum));
  // Update pointers and counter
  CurrentSrc = Builder.CreateGEP(I8Ty, CurrentSrc, Builder.getInt32(BytesNum));
  CurrentDst = Builder.CreateGEP(I8Ty, CurrentDst, Builder.getInt32(BytesNum));
}

// it supports 1-15 bytes
// src  unalign and dst  unalign
bool RISCVEsp32P4MemIntrinPass::processFromSrcUnalignDstUnalign1To15Const(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  IRBuilder<> Builder(M);
  Value *OrigSrc = M->getSource(); // Keep original pointers
  Value *OrigDst = M->getDest();
  Value *CurrentSrc = OrigSrc; // Pointers to use for copying
  Value *CurrentDst = OrigDst;

  ConstantInt *LenCI = dyn_cast<ConstantInt>(M->getLength());
  if (!LenCI)
    return false;
  uint64_t Len = LenCI->getZExtValue();

  assert(Len > 0 && Len < 16 && "Len must be between 1 and 15");

  Type *I8Ty = Builder.getInt8Ty();
  Type *I16Ty = Builder.getInt16Ty();
  Type *I32Ty = Builder.getInt32Ty();
  Type *I64Ty = Builder.getInt64Ty();
  Type *I32PtrTy = Builder.getInt32Ty(); // Type for asm operands

  uint64_t BytesCopied = 0;

  // Handle remaining 8 bytes
  if (Len - BytesCopied >= 8) {
    handleRemainingBytes(Builder, I64Ty, I8Ty, CurrentSrc, CurrentDst, 8);
    BytesCopied += 8;
  }

  // --- Use LLVM IR to handle remaining bytes (Len - BytesCopied) ---

  // Handle remaining 4 bytes
  if (Len - BytesCopied >= 4) {
    handleRemainingBytes(Builder, I32Ty, I8Ty, CurrentSrc, CurrentDst, 4);
    BytesCopied += 4;
  }

  // Handle remaining 2 bytes
  if (Len - BytesCopied >= 2) {
    handleRemainingBytes(Builder, I16Ty, I8Ty, CurrentSrc, CurrentDst, 2);
    BytesCopied += 2;
  }

  // Handle remaining 1 byte
  if (Len - BytesCopied >= 1) {
    Value *LoadVal = Builder.CreateAlignedLoad(I8Ty, CurrentSrc, Align(1));
    Builder.CreateAlignedStore(LoadVal, CurrentDst, Align(1));
    BytesCopied += 1;
  }

  // Remove the original memcpy instruction
  M->eraseFromParent();
  return true;
}

// Split len into multiples of 16 and Remainder
bool RISCVEsp32P4MemIntrinPass::processOtherConstAlign(MemCpyInst *M,
                                                       BasicBlock::iterator &BI,
                                                       uint64_t dstAlign,
                                                       uint64_t srcAlign) {
  uint64_t Remainder = Len % 16;
  uint64_t mainSize = Len - Remainder;

  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();
  Builder.CreateMemCpy(Dst, Align(dstAlign), Src, Align(srcAlign), mainSize);

  Value *NewSrc =
      Builder.CreateGEP(Builder.getInt8Ty(), Src, Builder.getInt64(mainSize));
  Value *NewDst =
      Builder.CreateGEP(Builder.getInt8Ty(), Dst, Builder.getInt64(mainSize));

  Builder.CreateMemCpy(NewDst, Align(dstAlign), NewSrc, Align(srcAlign),
                       Remainder);

  M->eraseFromParent();
  return true;
}

// src 16-byte aligned, dst 16-byte aligned, size is other constant
bool RISCVEsp32P4MemIntrinPass::processSrc16Dst16OtherConst(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processOtherConstAlign(M, BBI, 16, 16);
}

// src 16-byte aligned, dst 8-byte aligned, size is other constant
bool RISCVEsp32P4MemIntrinPass::processSrc16Dst8OtherConst(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processOtherConstAlign(M, BBI, 8, 16);
}

// src 8-byte aligned, dst 8-byte aligned, size is other constant
bool RISCVEsp32P4MemIntrinPass::processSrc8Dst8OtherConst(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processOtherConstAlign(M, BBI, 8, 8);
}

// src 8-byte aligned, dst 16-byte aligned, size is other constant
bool RISCVEsp32P4MemIntrinPass::processSrc8Dst16OtherConst(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processOtherConstAlign(M, BBI, 16, 8);
}

// src unalign, dst 16-byte aligned, size is other constant
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16OtherConst(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processOtherConstAlign(M, BBI, 16, M->getSourceAlign()->value());
}

// src unalign, dst 8-byte aligned, size is other constant
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst8OtherConst(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processOtherConstAlign(M, BBI, 8, M->getSourceAlign()->value());
}

// src 16| 8 align, dst 16|8 align , var 1-15
void RISCVEsp32P4MemIntrinPass::processSrc16Dst16From1To15Var(
    IRBuilder<> &Builder, Value *Dst, Value *Src, Value *Size, bool isInline,
    MemCpyType Type) {
  processMemCpyVarFrom1To15(Builder, "esp32p4MemCpySrc16Dst16From0To15Opt", Dst,
                            Src, Size, isInline, Type);
}

void RISCVEsp32P4MemIntrinPass::processMemCpyVarFrom1To15(
    IRBuilder<> &Builder, const std::string &FuncName, Value *Dst, Value *Src,
    Value *Size, bool isInline, MemCpyType Type) {

  LLVMContext &Ctx = Builder.getContext();
  // Check if the function already exists in the current TheModule
  if (useExistingHelperFunction(Builder, FuncName, Dst, Src, Size)) {
    return;
  }

  // Create the helper function signature (ptr, ptr, i32) -> void
  Function *MemCpyFunc = createMemCpyHelperFunctionPtr(Builder, FuncName, Dst,
                                                       Src, Size, isInline);

  // Create basic blocks for the helper function
  BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", MemCpyFunc);
  BasicBlock *ReturnBB = BasicBlock::Create(Ctx, "return",
                                            MemCpyFunc); // Common exit block

  // Create basic blocks for each case in the switch
  std::vector<BasicBlock *> SwitchBBs;
  for (int I = 1; I <= 15; I++) {
    SwitchBBs.push_back(
        BasicBlock::Create(Ctx, "sw.bb" + std::to_string(I), MemCpyFunc));
  }

  // --- Populate Entry Basic Block ---
  IRBuilder<> FuncBuilder(EntryBB);

  // Get function parameters
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");
  Value *SizeArg = MemCpyFunc->arg_begin() + 2;
  SizeArg->setName("size");

  // Create switch statement in the entry block
  SwitchInst *SI =
      FuncBuilder.CreateSwitch(SizeArg, ReturnBB, 15); // Default to return
  for (int I = 1; I <= 15; I++) {
    // Add case: if SizeArg == i, jump to SwitchBBs[i-1]
    SI->addCase(FuncBuilder.getInt32(I), SwitchBBs[I - 1]);
  }

  // --- Populate Switch Case Basic Blocks with LLVM IR ---
  llvm::Type *I8Ty = FuncBuilder.getInt8Ty();
  llvm::Type *I16Ty = FuncBuilder.getInt16Ty();
  llvm::Type *I32Ty = FuncBuilder.getInt32Ty();

  for (int I = 1; I <= 15; I++) {
    FuncBuilder.SetInsertPoint(
        SwitchBBs[I - 1]); // Set builder to the correct case block
    uint64_t BytesToCopy = I;
    uint64_t BytesCopied = 0;
    if (Type == MemCpyType::Src16_Dst16_Var ||
        Type == MemCpyType::Src16_Dst8_Var ||
        Type == MemCpyType::Src8_Dst16_Var ||
        Type == MemCpyType::Src8_Dst8_Var) {
      if (BytesToCopy - BytesCopied >= 8) {
        // Load and store 64-bit data (ptr passed directly)
        auto [VectorData, UpdatedSrc] = createEspVldL64Ip(FuncBuilder, SrcArg);
        DstArg = createEspVstL64Ip(FuncBuilder, VectorData, DstArg);
        SrcArg = UpdatedSrc;
        BytesCopied += 8;
      }
    }
    // Generate load/store sequence for copying 'i' bytes
    // Prioritize 4-byte copies
    while (BytesToCopy - BytesCopied >= 4) {
      Value *SrcOffset = FuncBuilder.getInt32(BytesCopied);
      Value *DstOffset = FuncBuilder.getInt32(BytesCopied);
      Value *SrcPtr =
          FuncBuilder.CreateGEP(I8Ty, SrcArg, SrcOffset, "src.gep.i32");
      Value *DstPtr =
          FuncBuilder.CreateGEP(I8Ty, DstArg, DstOffset, "dst.gep.i32");
      // Use natural alignment for the types
      Value *LoadVal = FuncBuilder.CreateAlignedLoad(I32Ty, SrcPtr, Align(4));
      FuncBuilder.CreateAlignedStore(LoadVal, DstPtr, Align(4));
      BytesCopied += 4;
    }
    // Handle remaining 2 bytes
    if (BytesToCopy - BytesCopied >= 2) {
      Value *SrcOffset = FuncBuilder.getInt32(BytesCopied);
      Value *DstOffset = FuncBuilder.getInt32(BytesCopied);
      Value *SrcPtr =
          FuncBuilder.CreateGEP(I8Ty, SrcArg, SrcOffset, "src.gep.i16");
      Value *DstPtr =
          FuncBuilder.CreateGEP(I8Ty, DstArg, DstOffset, "dst.gep.i16");
      Value *LoadVal = FuncBuilder.CreateAlignedLoad(I16Ty, SrcPtr, Align(2));
      FuncBuilder.CreateAlignedStore(LoadVal, DstPtr, Align(2));
      BytesCopied += 2;
    }
    // Handle remaining 1 byte
    if (BytesToCopy - BytesCopied >= 1) {
      Value *SrcOffset = FuncBuilder.getInt32(BytesCopied);
      Value *DstOffset = FuncBuilder.getInt32(BytesCopied);
      Value *SrcPtr =
          FuncBuilder.CreateGEP(I8Ty, SrcArg, SrcOffset, "src.gep.i8");
      Value *DstPtr =
          FuncBuilder.CreateGEP(I8Ty, DstArg, DstOffset, "dst.gep.i8");
      Value *LoadVal = FuncBuilder.CreateAlignedLoad(I8Ty, SrcPtr, Align(1));
      FuncBuilder.CreateAlignedStore(LoadVal, DstPtr, Align(1));
      BytesCopied += 1;
    }

    // After copying, branch to the common return block
    FuncBuilder.CreateBr(ReturnBB);
  }

  // --- Populate Return Basic Block ---
  FuncBuilder.SetInsertPoint(ReturnBB);
  FuncBuilder.CreateRetVoid(); // Add return void instruction

  return;
}

void RISCVEsp32P4MemIntrinPass::processMemCpyVarFrom1To7(
    IRBuilder<> &Builder, const std::string &FuncName, Value *Dst, Value *Src,
    Value *Size, bool isInline) {

  LLVMContext &Ctx = Builder.getContext();
  // Check if the function already exists in the current TheModule
  if (useExistingHelperFunction(Builder, FuncName, Dst, Src, Size)) {
    return;
  }

  // Create the helper function signature (ptr, ptr, i32) -> void
  Function *MemCpyFunc = createMemCpyHelperFunctionPtr(Builder, FuncName, Dst,
                                                       Src, Size, isInline);

  // Create basic blocks for the helper function
  BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", MemCpyFunc);
  BasicBlock *ReturnBB = BasicBlock::Create(Ctx, "return",
                                            MemCpyFunc); // Common exit block

  // Create basic blocks for each case in the switch
  std::vector<BasicBlock *> SwitchBBs;
  for (int I = 1; I <= 7; I++) {
    SwitchBBs.push_back(
        BasicBlock::Create(Ctx, "sw.bb" + std::to_string(I), MemCpyFunc));
  }

  // --- Populate Entry Basic Block ---
  IRBuilder<> FuncBuilder(EntryBB);

  // Get function parameters
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");
  Value *SizeArg = MemCpyFunc->arg_begin() + 2;
  SizeArg->setName("size");

  // Create switch statement in the entry block
  SwitchInst *SI =
      FuncBuilder.CreateSwitch(SizeArg, ReturnBB, 7); // Default to return
  for (int I = 1; I <= 7; I++) {
    // Add case: if SizeArg == i, jump to SwitchBBs[i-1]
    SI->addCase(FuncBuilder.getInt32(I), SwitchBBs[I - 1]);
  }

  // --- Populate Switch Case Basic Blocks with LLVM IR ---
  llvm::Type *I8Ty = FuncBuilder.getInt8Ty();
  llvm::Type *I16Ty = FuncBuilder.getInt16Ty();
  llvm::Type *I32Ty = FuncBuilder.getInt32Ty();

  for (int I = 1; I <= 7; I++) {
    FuncBuilder.SetInsertPoint(
        SwitchBBs[I - 1]); // Set builder to the correct case block
    uint64_t BytesToCopy = I;
    uint64_t BytesCopied = 0;
    // Generate load/store sequence for copying 'i' bytes
    // Prioritize 4-byte copies
    while (BytesToCopy - BytesCopied >= 4) {
      Value *SrcOffset = FuncBuilder.getInt32(BytesCopied);
      Value *DstOffset = FuncBuilder.getInt32(BytesCopied);
      Value *SrcPtr =
          FuncBuilder.CreateGEP(I8Ty, SrcArg, SrcOffset, "src.gep.i32");
      Value *DstPtr =
          FuncBuilder.CreateGEP(I8Ty, DstArg, DstOffset, "dst.gep.i32");
      // Use natural alignment for the types
      Value *LoadVal = FuncBuilder.CreateAlignedLoad(I32Ty, SrcPtr, Align(4));
      FuncBuilder.CreateAlignedStore(LoadVal, DstPtr, Align(4));
      BytesCopied += 4;
    }
    // Handle remaining 2 bytes
    if (BytesToCopy - BytesCopied >= 2) {
      Value *SrcOffset = FuncBuilder.getInt32(BytesCopied);
      Value *DstOffset = FuncBuilder.getInt32(BytesCopied);
      Value *SrcPtr =
          FuncBuilder.CreateGEP(I8Ty, SrcArg, SrcOffset, "src.gep.i16");
      Value *DstPtr =
          FuncBuilder.CreateGEP(I8Ty, DstArg, DstOffset, "dst.gep.i16");
      Value *LoadVal = FuncBuilder.CreateAlignedLoad(I16Ty, SrcPtr, Align(2));
      FuncBuilder.CreateAlignedStore(LoadVal, DstPtr, Align(2));
      BytesCopied += 2;
    }
    // Handle remaining 1 byte
    if (BytesToCopy - BytesCopied >= 1) {
      Value *SrcOffset = FuncBuilder.getInt32(BytesCopied);
      Value *DstOffset = FuncBuilder.getInt32(BytesCopied);
      Value *SrcPtr =
          FuncBuilder.CreateGEP(I8Ty, SrcArg, SrcOffset, "src.gep.i8");
      Value *DstPtr =
          FuncBuilder.CreateGEP(I8Ty, DstArg, DstOffset, "dst.gep.i8");
      Value *LoadVal = FuncBuilder.CreateAlignedLoad(I8Ty, SrcPtr, Align(1));
      FuncBuilder.CreateAlignedStore(LoadVal, DstPtr, Align(1));
      BytesCopied += 1;
    }

    // After copying, branch to the common return block
    FuncBuilder.CreateBr(ReturnBB);
  }

  // --- Populate Return Basic Block ---
  FuncBuilder.SetInsertPoint(ReturnBB);
  FuncBuilder.CreateRetVoid(); // Add return void instruction

  return;
}

bool RISCVEsp32P4MemIntrinPass::processSrc16Dst16Var(MemCpyInst *M) {
  return processMemCpyWithAlignmentVar(M, "Src16Dst16", 16, 16);
}

bool RISCVEsp32P4MemIntrinPass::processMemCpyWithAlignmentVar(
    MemCpyInst *M, std::string srcdstcase, unsigned SrcAlign,
    unsigned DstAlign) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();
  Value *Size = M->getLength();
  std::string FuncName = "esp32p4MemCpy" + srcdstcase + "Var";

  if (useExistingHelperFunction(M, Builder, FuncName, Dst, Src, Size)) {
    return true;
  }

  Function *MemCpyFunc =
      createMemCpyHelperFunctionPtr(Builder, FuncName, Dst, Src, Size, false);

  Value *DstArg = MemCpyFunc->arg_begin();
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  Value *SizeArg = MemCpyFunc->arg_begin() + 2;
  Value *DstArgOrg = DstArg;
  Value *SrcArgOrg = SrcArg;

  BasicBlock *EntryBB =
      BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
  BasicBlock *HandleSmallSize =
      BasicBlock::Create(M->getContext(), "handle.small.size", MemCpyFunc);
  BasicBlock *CheckMidSizeRange =
      BasicBlock::Create(M->getContext(), "check.mid.range", MemCpyFunc);
  BasicBlock *HandleMidSize =
      BasicBlock::Create(M->getContext(), "handle.mid.size", MemCpyFunc);
  BasicBlock *HandleLargeSizeLoop =
      BasicBlock::Create(M->getContext(), "handle.large.loop", MemCpyFunc);
  BasicBlock *ReturnBB =
      BasicBlock::Create(M->getContext(), "return", MemCpyFunc);

  IRBuilder<> FuncBuilder(EntryBB);
  Value *IsLT8 =
      FuncBuilder.CreateICmpULT(SizeArg, FuncBuilder.getInt32(8), "is.lt.8");
  FuncBuilder.CreateCondBr(IsLT8, HandleSmallSize, CheckMidSizeRange);

  FuncBuilder.SetInsertPoint(HandleSmallSize);
  std::string FuncName1_7 = "esp32p4MemCpy" + srcdstcase + "From1To7Opt";
  processMemCpyVarFrom1To7(FuncBuilder, FuncName1_7, DstArg, SrcArg, SizeArg,
                           false);
  FuncBuilder.CreateBr(ReturnBB);

  FuncBuilder.SetInsertPoint(CheckMidSizeRange);
  Value *IsLT16 =
      FuncBuilder.CreateICmpULT(SizeArg, FuncBuilder.getInt32(16), "is.lt.16");
  FuncBuilder.CreateCondBr(IsLT16, HandleMidSize, HandleLargeSizeLoop);

  FuncBuilder.SetInsertPoint(HandleMidSize);
  auto [VectorData, UpdatedSrc] = createEspVldL64Ip(FuncBuilder, SrcArg);
  SrcArg = UpdatedSrc;
  DstArg = createEspVstL64Ip(FuncBuilder, VectorData, DstArg);
  Value *SizeMinus8 = FuncBuilder.CreateAdd(SizeArg, FuncBuilder.getInt32(-8),
                                            "size.minus.8", false, true);
  processMemCpyVarFrom1To7(FuncBuilder, FuncName1_7, DstArg, SrcArg, SizeMinus8,
                           true);
  FuncBuilder.CreateBr(ReturnBB);

  FuncBuilder.SetInsertPoint(HandleLargeSizeLoop);
  Value *Num128BBlocks = FuncBuilder.CreateLShr(
      SizeArg, FuncBuilder.getInt32(7), "num.128B.blocks");
  Value *Num16BBlocks = FuncBuilder.CreateLShr(SizeArg, FuncBuilder.getInt32(4),
                                               "num.16B.blocks");
  Value *Remaining16B = FuncBuilder.CreateAnd(
      Num16BBlocks, FuncBuilder.getInt32(7), "remaining.16B.blocks");
  Value *RemainingBytes = FuncBuilder.CreateAnd(
      SizeArg, FuncBuilder.getInt32(7), "remaining.bytes");
  Value *IsSmallSize128 = FuncBuilder.CreateICmpULT(
      SizeArg, FuncBuilder.getInt32(128), "is.lt.128");

  BasicBlock *LoopExitCleanup =
      BasicBlock::Create(M->getContext(), "loop.exit.cleanup", MemCpyFunc);
  BasicBlock *LoopBody128B =
      BasicBlock::Create(M->getContext(), "loop.body.128B", MemCpyFunc);
  BasicBlock *HandleTailBlockSwitch =
      BasicBlock::Create(M->getContext(), "handle.tail.switch", MemCpyFunc);
  BasicBlock *InvalidCaseTrap =
      BasicBlock::Create(M->getContext(), "invalid.switch.trap", MemCpyFunc);

  FuncBuilder.CreateCondBr(IsSmallSize128, LoopExitCleanup, LoopBody128B);

  FuncBuilder.SetInsertPoint(InvalidCaseTrap);
  FuncBuilder.CreateUnreachable();

  FuncBuilder.SetInsertPoint(LoopBody128B);
  PHINode *LoopIndex =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "loop.index");
  LoopIndex->addIncoming(FuncBuilder.getInt32(0), HandleLargeSizeLoop);
  PHINode *SrcPtrInLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getPtrTy(), 2, "src.ptr.loop");
  SrcPtrInLoop->addIncoming(SrcArgOrg, HandleLargeSizeLoop);
  PHINode *DstPtrInLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getPtrTy(), 2, "dst.ptr.loop");
  DstPtrInLoop->addIncoming(DstArgOrg, HandleLargeSizeLoop);

  Value *CurrentSrc = SrcPtrInLoop;
  Value *CurrentDst = DstPtrInLoop;
  for (int BatchStart = 0; BatchStart < 8; BatchStart += 4) {
    SmallVector<std::pair<Value *, Value *>, 4> LoadedVectorPairs;
    for (int I = 0; I < 4; I++) {
      if (SrcAlign == 16 && DstAlign == 16) {
        auto [Data, UpdatedPtr] = createEspVld128Ip(FuncBuilder, CurrentSrc);
        LoadedVectorPairs.push_back({Data, nullptr});
        CurrentSrc = UpdatedPtr;
      } else {
        auto [L64Data, L64Ptr] = createEspVldL64Ip(FuncBuilder, CurrentSrc);
        auto [H64Data, H64Ptr] = createEspVldH64Ip(FuncBuilder, L64Ptr);
        LoadedVectorPairs.push_back({L64Data, H64Data});
        CurrentSrc = H64Ptr;
      }
    }

    for (auto [FirstData, SecondData] : LoadedVectorPairs) {
      if (SecondData == nullptr) {
        if (DstAlign == 16) {
          CurrentDst = createEspVst128Ip(FuncBuilder, FirstData, CurrentDst);
        } else {
          Value *L64Data = FuncBuilder.CreateShuffleVector(
              FirstData, PoisonValue::get(FirstData->getType()),
              ArrayRef<int>{0, 1, 2, 3, 4, 5, 6, 7});
          Value *H64Data = FuncBuilder.CreateShuffleVector(
              FirstData, PoisonValue::get(FirstData->getType()),
              ArrayRef<int>{8, 9, 10, 11, 12, 13, 14, 15});
          Value *AfterL64 = createEspVstL64Ip(FuncBuilder, L64Data, CurrentDst);
          CurrentDst = createEspVstH64Ip(FuncBuilder, H64Data, AfterL64);
        }
      } else {
        Value *AfterL64 = createEspVstL64Ip(FuncBuilder, FirstData, CurrentDst);
        CurrentDst = createEspVstH64Ip(FuncBuilder, SecondData, AfterL64);
      }
    }
  }
  SrcPtrInLoop->addIncoming(CurrentSrc, LoopBody128B);
  DstPtrInLoop->addIncoming(CurrentDst, LoopBody128B);

  Value *LoopNext = FuncBuilder.CreateAdd(LoopIndex, FuncBuilder.getInt32(1),
                                          "loop.inc", true, true);
  LoopIndex->addIncoming(LoopNext, LoopBody128B);
  Value *IsLoopDone =
      FuncBuilder.CreateICmpEQ(LoopNext, Num128BBlocks, "loop.done");
  FuncBuilder.CreateCondBr(IsLoopDone, LoopExitCleanup, LoopBody128B);

  FuncBuilder.SetInsertPoint(LoopExitCleanup);
  PHINode *SrcPtrAfterLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getPtrTy(), 2, "src.ptr.after.loop");
  SrcPtrAfterLoop->addIncoming(SrcArgOrg, HandleLargeSizeLoop);
  // Use CurrentSrc (which is %20 in .ll) instead of SrcPtrInLoop
  SrcPtrAfterLoop->addIncoming(CurrentSrc, LoopBody128B);
  PHINode *DstPtrAfterLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getPtrTy(), 2, "dst.ptr.after.loop");
  DstPtrAfterLoop->addIncoming(DstArgOrg, HandleLargeSizeLoop);
  // Use CurrentDst (which is %vst128ip_m14 in .ll) instead of DstPtrInLoop
  DstPtrAfterLoop->addIncoming(CurrentDst, LoopBody128B);

  SwitchInst *Switch = FuncBuilder.CreateSwitch(Remaining16B, InvalidCaseTrap);
  FuncBuilder.SetInsertPoint(HandleTailBlockSwitch);

  PHINode *SrcPtrInTailSwitch =
      FuncBuilder.CreatePHI(FuncBuilder.getPtrTy(), 2, "src.ptr.tail");
  SrcPtrInTailSwitch->addIncoming(SrcPtrAfterLoop, LoopExitCleanup);
  PHINode *DstPtrInTailSwitch =
      FuncBuilder.CreatePHI(FuncBuilder.getPtrTy(), 2, "dst.ptr.tail");
  DstPtrInTailSwitch->addIncoming(DstPtrAfterLoop, LoopExitCleanup);

  BasicBlock *Handle8ByteTail =
      BasicBlock::Create(M->getContext(), "handle.8B.tail", MemCpyFunc);
  BasicBlock *After8ByteTail =
      BasicBlock::Create(M->getContext(), "after.8B.tail", MemCpyFunc);

  Value *Has8ByteTail = FuncBuilder.CreateICmpEQ(
      FuncBuilder.CreateAnd(SizeArg, FuncBuilder.getInt32(8)),
      FuncBuilder.getInt32(0));
  FuncBuilder.CreateCondBr(Has8ByteTail, After8ByteTail, Handle8ByteTail);

  for (int I = 1; I <= 7; I++) {
    BasicBlock *CaseBB = BasicBlock::Create(
        M->getContext(), "tail.case." + std::to_string(I), MemCpyFunc);
    Switch->addCase(FuncBuilder.getInt32(I), CaseBB);
    FuncBuilder.SetInsertPoint(CaseBB);

    Value *CurrentSrc = SrcPtrAfterLoop;
    Value *CurrentDst = DstPtrAfterLoop;

    for (int BatchStart = 0; BatchStart < I; BatchStart += 4) {
      int BatchSize = I - BatchStart;
      if (BatchSize > 4)
        BatchSize = 4;

      SmallVector<std::pair<Value *, Value *>, 4> TailVectorPairs;
      for (int J = 0; J < BatchSize; J++) {
        if (SrcAlign == 16) {
          auto [Data, UpdatedPtr] = createEspVld128Ip(FuncBuilder, CurrentSrc);
          TailVectorPairs.push_back({Data, nullptr});
          CurrentSrc = UpdatedPtr;
        } else {
          auto [L64Data, L64Ptr] = createEspVldL64Ip(FuncBuilder, CurrentSrc);
          auto [H64Data, H64Ptr] = createEspVldH64Ip(FuncBuilder, L64Ptr);
          TailVectorPairs.push_back({L64Data, H64Data});
          CurrentSrc = H64Ptr;
        }
      }

      for (auto [FirstData, SecondData] : TailVectorPairs) {
        if (SecondData == nullptr) {
          if (DstAlign == 16) {
            CurrentDst = createEspVst128Ip(FuncBuilder, FirstData, CurrentDst);
          } else {
            Value *L64Data = FuncBuilder.CreateShuffleVector(
                FirstData, PoisonValue::get(FirstData->getType()),
                ArrayRef<int>{0, 1, 2, 3, 4, 5, 6, 7});
            Value *H64Data = FuncBuilder.CreateShuffleVector(
                FirstData, PoisonValue::get(FirstData->getType()),
                ArrayRef<int>{8, 9, 10, 11, 12, 13, 14, 15});
            Value *AfterL64 =
                createEspVstL64Ip(FuncBuilder, L64Data, CurrentDst);
            CurrentDst = createEspVstH64Ip(FuncBuilder, H64Data, AfterL64);
          }
        } else {
          Value *AfterL64 =
              createEspVstL64Ip(FuncBuilder, FirstData, CurrentDst);
          CurrentDst = createEspVstH64Ip(FuncBuilder, SecondData, AfterL64);
        }
      }
    }

    SrcPtrInTailSwitch->addIncoming(CurrentSrc, CaseBB);
    DstPtrInTailSwitch->addIncoming(CurrentDst, CaseBB);
    FuncBuilder.CreateBr(HandleTailBlockSwitch);
  }
  Switch->addCase(FuncBuilder.getInt32(0), HandleTailBlockSwitch);

  FuncBuilder.SetInsertPoint(Handle8ByteTail);
  auto [TailVectorData, TailUpdatedSrc] =
      createEspVldL64Ip(FuncBuilder, SrcPtrInTailSwitch);
  SrcArg = TailUpdatedSrc;
  DstArg = createEspVstL64Ip(FuncBuilder, TailVectorData, DstPtrInTailSwitch);
  FuncBuilder.CreateBr(After8ByteTail);

  FuncBuilder.SetInsertPoint(After8ByteTail);
  PHINode *SrcPtrAfter8B =
      FuncBuilder.CreatePHI(FuncBuilder.getPtrTy(), 2, "src.ptr.after.8B");
  SrcPtrAfter8B->addIncoming(SrcPtrInTailSwitch, HandleTailBlockSwitch);
  SrcPtrAfter8B->addIncoming(SrcArg, Handle8ByteTail);
  PHINode *DstPtrAfter8B =
      FuncBuilder.CreatePHI(FuncBuilder.getPtrTy(), 2, "dst.ptr.after.8B");
  DstPtrAfter8B->addIncoming(DstPtrInTailSwitch, HandleTailBlockSwitch);
  DstPtrAfter8B->addIncoming(DstArg, Handle8ByteTail);
  Value *HasRemainingBytes =
      FuncBuilder.CreateICmpEQ(RemainingBytes, FuncBuilder.getInt32(0));
  BasicBlock *HandleRemainingBytes =
      BasicBlock::Create(M->getContext(), "handle.remaining.bytes", MemCpyFunc);
  FuncBuilder.CreateCondBr(HasRemainingBytes, ReturnBB, HandleRemainingBytes);

  FuncBuilder.SetInsertPoint(HandleRemainingBytes);
  processMemCpyVarFrom1To7(FuncBuilder, FuncName1_7, DstPtrAfter8B,
                           SrcPtrAfter8B, RemainingBytes, true);
  FuncBuilder.CreateBr(ReturnBB);

  FuncBuilder.SetInsertPoint(ReturnBB);
  FuncBuilder.CreateRetVoid();

  M->eraseFromParent();
  return true;
}

bool RISCVEsp32P4MemIntrinPass::processSrc16Dst8Var(MemCpyInst *M) {
  return processMemCpyWithAlignmentVar(M, "Src16Dst8", 16, 8);
}

bool RISCVEsp32P4MemIntrinPass::processSrc8Dst16Var(MemCpyInst *M) {
  return processMemCpyWithAlignmentVar(M, "Src8Dst16", 8, 16);
}

bool RISCVEsp32P4MemIntrinPass::processSrc8Dst8Var(MemCpyInst *M) {
  return processMemCpyWithAlignmentVar(M, "Src8Dst8", 8, 8);
}

bool RISCVEsp32P4MemIntrinPass::processSrc16DstUnalignVar(
    MemCpyInst *M, BasicBlock::iterator &BBI) {

  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();
  Value *Size = M->getLength();

  std::string FuncName = "esp32p4MemCpySrc16DstunalignVar";

  if (Function *ExistingFunc = TheModule->getFunction(FuncName)) {
    Builder.CreateCall(ExistingFunc,
                       {Dst, Src, Size, Builder.getInt32(DstAlignValue)});
    M->eraseFromParent();
    return true;
  }

  FunctionType *FuncTy =
      FunctionType::get(Builder.getVoidTy(),
                        {Builder.getPtrTy(), Builder.getPtrTy(),
                         Builder.getInt32Ty(), Builder.getInt32Ty()},
                        false);

  Function *HelperFunc = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                          FuncName, M->getModule());

  Value *CallArgs[] = {Dst, Src, Size, Builder.getInt32(DstAlignValue)};
  CallInst *TailCall = CallInst::Create(HelperFunc->getFunctionType(),
                                        HelperFunc, CallArgs, "", nullptr);
  TailCall->setTailCallKind(CallInst::TCK_Tail);
  Builder.Insert(TailCall);

  HelperFunc->addFnAttr(Attribute::NoUnwind);
  HelperFunc->addFnAttr(Attribute::NoInline);

  BasicBlock *EntryBB =
      BasicBlock::Create(M->getContext(), "entry", HelperFunc);
  BasicBlock *HandleAlignedHead =
      BasicBlock::Create(M->getContext(), "handle.head", HelperFunc);
  BasicBlock *HandleAlignedTail =
      BasicBlock::Create(M->getContext(), "handle.tail", HelperFunc);
  BasicBlock *ReturnBB =
      BasicBlock::Create(M->getContext(), "return", HelperFunc);

  IRBuilder<> FuncBuilder(EntryBB);

  auto ArgIter = HelperFunc->arg_begin();
  Value *DstArg = ArgIter++;
  DstArg->setName("dst");
  Value *SrcArg = ArgIter++;
  SrcArg->setName("src");
  Value *SizeArg = ArgIter++;
  SizeArg->setName("size");
  Value *DstAlignArg = ArgIter++;
  DstAlignArg->setName("dst_align");

  Value *HeadSize =
      FuncBuilder.CreateSub(FuncBuilder.getInt32(16), DstAlignArg, "head.size");

  Value *NeedSplit = FuncBuilder.CreateICmpULT(HeadSize, SizeArg, "need.split");
  FuncBuilder.CreateCondBr(NeedSplit, HandleAlignedTail, HandleAlignedHead);

  FuncBuilder.SetInsertPoint(HandleAlignedHead);
  processMemCpyVarFrom1To15(
      FuncBuilder, "esp32p4MemCpySrcUnalignDstUnalignFrom1To15Opt", DstArg,
      SrcArg, SizeArg,
      /*isDstUnaligned*/ true, MemCpyType::SrcUnalign_DstUnalign_Var);
  FuncBuilder.CreateBr(ReturnBB);

  FuncBuilder.SetInsertPoint(HandleAlignedTail);

  processMemCpyVarFrom1To15(
      FuncBuilder, "esp32p4MemCpySrcUnalignDstUnalignFrom1To15Opt", DstArg,
      SrcArg, HeadSize,
      /*isDstUnaligned*/ true, MemCpyType::SrcUnalign_DstUnalign_Var);

  Value *RemainingBytes =
      FuncBuilder.CreateSub(SizeArg, HeadSize, "remaining.size");

  Value *DstAlignedPtr = FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), DstArg,
                                               HeadSize, "dst.aligned.ptr");
  Value *SrcAlignedPtr = FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), SrcArg,
                                               HeadSize, "src.aligned.ptr");

  // Use the standard LLVM MemCpy Intrinsic to copy the remaining part (src
  // 16-byte aligned, dst unaligned)
  FuncBuilder.CreateMemCpy(DstAlignedPtr, Align(16), SrcAlignedPtr, Align(1),
                           RemainingBytes);

  FuncBuilder.CreateBr(ReturnBB);

  // --- [3] return block
  FuncBuilder.SetInsertPoint(ReturnBB);
  FuncBuilder.CreateRetVoid();

  M->eraseFromParent();
  return true;
}

// src 16-byte aligned, dst is unalign, size is other constant
bool RISCVEsp32P4MemIntrinPass::processSrc16DstUnalignOtherConst(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrc16DstUnalignVar(M, BBI);
}

// src 16-byte aligned, dst is unalign, size is divisible by 8
bool RISCVEsp32P4MemIntrinPass::processSrc16DstUnalignConst8(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrc16DstUnalignVar(M, BBI);
}

// src 16-byte aligned, dst is unalign, size is divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrc16DstUnalignConst16(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrc16DstUnalignVar(M, BBI);
}

// src 8-byte aligned, dst unaligned, size divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrc8DstUnalignConst16(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrc16DstUnalignVar(M, BBI);
}

// src 8-byte aligned, dst unaligned, size divisible by 8
bool RISCVEsp32P4MemIntrinPass::processSrc8DstUnalignConst8(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrc16DstUnalignVar(M, BBI);
}

// src 8-byte aligned, dst unaligned, size is other constant
bool RISCVEsp32P4MemIntrinPass::processSrc8DstUnalignOtherConst(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrc16DstUnalignVar(M, BBI);
}

// src 8-byte aligned, dst unaligned, size is variable
bool RISCVEsp32P4MemIntrinPass::processSrc8DstUnalignVar(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrc16DstUnalignVar(M, BBI);
}

// Return the updated src pointer (i32)
std::tuple<Value *, Value *, Value *>
RISCVEsp32P4MemIntrin::createEspLd128UsarIp(IRBuilder<> &Builder, Value *Src) {
  Type *i32Ty = Builder.getInt32Ty();
  Type *PtrTy = Builder.getPtrTy();
  Value *SrcPtr =
      Src->getType()->isPointerTy() ? Src : Builder.CreateIntToPtr(Src, PtrTy);

  Function *IntrinsicFunc = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_ld_128_usar_ip_m, {});
  Value *Call = Builder.CreateCall(
      IntrinsicFunc, {SrcPtr, ConstantInt::get(i32Ty, 16)}, "ld128usarip_m");

  Value *VectorData = Builder.CreateExtractValue(Call, 0);
  Value *UpdatedPtr = Builder.CreateExtractValue(Call, 1);
  Value *SarBytes = Builder.CreateExtractValue(Call, 2);

  return {VectorData, UpdatedPtr, SarBytes};
}

// Legacy interface for backward compatibility during migration
Value *RISCVEsp32P4MemIntrin::createEspLd128UsarIp(IRBuilder<> &Builder,
                                                   Value *Src, int Index) {
  assert(Index >= 0 && Index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc = Intrinsic::getDeclarationIfExists(
      TheModule, Intrinsic::riscv_esp_ld_128_usar_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {Src, ConstantInt::get(i32Ty, 16), ConstantInt::get(i32Ty, Index)},
      "ld128usarip");
}

// Return the updated src pointer (i32)
std::tuple<Value *, Value *, Value *>
RISCVEsp32P4MemIntrin::createEspSrcQLdIp(IRBuilder<> &Builder, Value *SarBytes,
                                         Value *Qy, Value *Qw, Value *SrcPtr,
                                         int Imm) {
  Type *i32Ty = Builder.getInt32Ty();
  Type *PtrTy = Builder.getPtrTy();
  Value *Ptr = SrcPtr->getType()->isPointerTy()
                   ? SrcPtr
                   : Builder.CreateIntToPtr(SrcPtr, PtrTy);
  Function *IntrinsicFunc = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_src_q_ld_ip_m, {});
  Value *Call = Builder.CreateCall(
      IntrinsicFunc, {SarBytes, Qy, Qw, Ptr, ConstantInt::get(i32Ty, Imm)},
      "srcqldip_m");
  Value *QwNew = Builder.CreateExtractValue(Call, 0);
  Value *QuNew = Builder.CreateExtractValue(Call, 1);
  Value *UpdatedPtr = Builder.CreateExtractValue(Call, 2);
  return {QwNew, QuNew, UpdatedPtr};
}

Value *RISCVEsp32P4MemIntrin::createEspSrcQM(IRBuilder<> &Builder,
                                             Value *SarBytes, Value *Qy,
                                             Value *Qw) {
  Function *IntrinsicFunc = Intrinsic::getOrInsertDeclaration(
      TheModule, Intrinsic::riscv_esp_src_q_m, {});
  Value *Call = Builder.CreateCall(IntrinsicFunc, {SarBytes, Qy, Qw}, "srcq_m");
  return Call;
}

// No pointer returned, pure calculation instructions
void RISCVEsp32P4MemIntrin::createEspSrcQ(IRBuilder<> &Builder, int Index0,
                                          int Index1, int Index2) {
  assert(Index0 >= 0 && Index0 <= 7 && "Index must be between 0 and 7");
  assert(Index1 >= 0 && Index1 <= 7 && "Index must be between 0 and 7");
  assert(Index2 >= 0 && Index2 <= 7 && "Index must be between 0 and 7");
  Function *IntrinsicFunc =
      Intrinsic::getOrInsertDeclaration(TheModule, Intrinsic::riscv_esp_src_q);
  Type *i32Ty = Builder.getInt32Ty();
  // Intrinsic arguments: q_idx1, q_idx2, q_idx_dst
  Builder.CreateCall(IntrinsicFunc, {ConstantInt::get(i32Ty, Index2),
                                     ConstantInt::get(i32Ty, Index1),
                                     ConstantInt::get(i32Ty, Index0)});
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstDiv48(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t Quotient) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  static int FuncCounter = 0;
  std::string FuncName = "esp32p4MemCpySrcunalignedDst16Div48Index" +
                         std::to_string(FuncCounter++);

  Function *MemCpyFunc =
      createMemCpyHelperFunctionPtrNoSize(Builder, FuncName, Dst, Src, false);

  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");

  BasicBlock *EntryBB = nullptr, *ForBodyBB = nullptr,
             *ForCondCleanupBB = nullptr;
  createLoopBlocks(MemCpyFunc, EntryBB, ForBodyBB, ForCondCleanupBB);

  IRBuilder<> FuncBuilder(EntryBB);
  Type *PtrTy = FuncBuilder.getPtrTy();
  Value *SrcPtr = SrcArg;
  Value *DstPtr = DstArg;

  auto [VecData1, Ptr1, Sar1] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
  SrcPtr = Ptr1;
  auto [VecData2, Ptr2, Sar2] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
  SrcPtr = Ptr2;
  FuncBuilder.CreateBr(ForBodyBB);

  FuncBuilder.SetInsertPoint(ForCondCleanupBB);
  FuncBuilder.CreateRetVoid();

  FuncBuilder.SetInsertPoint(ForBodyBB);
  PHINode *LoopCounter =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "loop.counter");
  LoopCounter->addIncoming(FuncBuilder.getInt32(0), EntryBB);
  PHINode *SrcPtrInLoop = FuncBuilder.CreatePHI(PtrTy, 2, "src.ptr.in.loop");
  SrcPtrInLoop->addIncoming(SrcPtr, EntryBB);

  PHINode *DstPtrInLoop = FuncBuilder.CreatePHI(PtrTy, 2, "dst.ptr.in.loop");
  DstPtrInLoop->addIncoming(DstPtr, EntryBB);

  Type *V16I8 = VectorType::get(FuncBuilder.getInt8Ty(), 16, false);
  PHINode *V0 = FuncBuilder.CreatePHI(V16I8, 2, "v0");
  V0->addIncoming(VecData1, EntryBB);
  PHINode *V1 = FuncBuilder.CreatePHI(V16I8, 2, "v1");
  V1->addIncoming(VecData2, EntryBB);
  PHINode *V2 = FuncBuilder.CreatePHI(V16I8, 2, "v2");
  V2->addIncoming(PoisonValue::get(V16I8), EntryBB);

  Value *LoopCounterIncremented =
      FuncBuilder.CreateAdd(LoopCounter, FuncBuilder.getInt32(1),
                            "loop.counter.incremented", true, true);
  LoopCounter->addIncoming(LoopCounterIncremented, ForBodyBB);

  // Each block = shift(prev_qu, prev_qw): block2 needs (load3, load2), block3
  // needs (load4, load3).
  Value *DstPtrVar = DstPtrInLoop;
  auto [V2New, V1New, SrcArg1] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V1, V0, SrcPtrInLoop, 16);
  DstPtrVar = createEspVst128Ip(FuncBuilder, V2New, DstPtrVar);

  auto [V0New, V2New2, SrcArg2] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V1New, V1, SrcArg1, 16);
  DstPtrVar = createEspVst128Ip(FuncBuilder, V0New, DstPtrVar);

  // Step 3: (qy,qw)=(v0.next, v2.next)=(V2New2, V1New)
  auto [V1New2, V0New2, SrcArg3] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V2New2, V1New, SrcArg2, 16);
  DstPtrVar = createEspVst128Ip(FuncBuilder, V1New2, DstPtrVar);

  // Next iteration: v0=v0.next (step2 index1), v1=v1.next (step3 index1),
  // v2=v2.next (step1 index1).
  SrcPtrInLoop->addIncoming(SrcArg3, ForBodyBB);
  DstPtrInLoop->addIncoming(DstPtrVar, ForBodyBB);
  V0->addIncoming(V2New2, ForBodyBB);
  V1->addIncoming(V0New2, ForBodyBB);
  V2->addIncoming(V1New, ForBodyBB);

  Value *LoopCompleted = FuncBuilder.CreateICmpEQ(
      LoopCounterIncremented, FuncBuilder.getInt32(Quotient), "loop.completed");
  FuncBuilder.CreateCondBr(LoopCompleted, ForCondCleanupBB, ForBodyBB);

  M->eraseFromParent();

  return true;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstMod48From32To47(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t Quotient,
    uint64_t Remainder) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  static int FuncCounter = 0;
  std::string FuncName = "esp32p4MemCpySrcunalignedDst16mod48From32To47Index" +
                         std::to_string(FuncCounter++);

  Function *MemCpyFunc =
      createMemCpyHelperFunctionPtrNoSize(Builder, FuncName, Dst, Src, false);
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");

  IRBuilder<> FuncBuilder(MemCpyFunc->getContext());
  Type *PtrTy = FuncBuilder.getPtrTy();
  Value *DstPtr = DstArg;

  // Quotient==1: generate linear IR (no loop/PHI) to match expected pattern.
  if (Quotient == 1) {
    BasicBlock *EntryBB =
        BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
    FuncBuilder.SetInsertPoint(EntryBB);
    Value *SrcPtr = SrcArg;
    auto [VecData1, Ptr1, Sar1] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
    SrcPtr = Ptr1;
    auto [VecData2, Ptr2, Sar2] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
    SrcPtr = Ptr2;

    // Three src.q.ld.ip + vst: (qy,qw)=(v_high,v_low)=(VecData2,VecData1);
    // store first return (QwNew) each time.
    auto [V2NewL, V1NewL, SrcArg1L] =
        createEspSrcQLdIp(FuncBuilder, Sar2, VecData2, VecData1, SrcPtr, 16);
    DstPtr = createEspVst128Ip(FuncBuilder, V2NewL, DstPtr);
    auto [V0NewL, V2New2L, SrcArg2L] =
        createEspSrcQLdIp(FuncBuilder, Sar2, V1NewL, VecData2, SrcArg1L, 16);
    DstPtr = createEspVst128Ip(FuncBuilder, V0NewL, DstPtr);
    // Step 3: (qy,qw)=(v0_next, v2_loaded)=(step2 index1, step1
    // index1)=(V2New2L, V1NewL).
    auto [V1New2L, V0New2L, SrcArg3L] =
        createEspSrcQLdIp(FuncBuilder, Sar2, V2New2L, V1NewL, SrcArg2L, 16);
    DstPtr = createEspVst128Ip(FuncBuilder, V1New2L, DstPtr);

    // Tail: (qy,qw)=(v1_next, v0_next)=(step3 index1, step2 index1)=(V0New2L,
    // V2New2L).
    auto [V2TailL, V1TailL, SrcArgTailL] =
        createEspSrcQLdIp(FuncBuilder, Sar2, V0New2L, V2New2L, SrcArg3L, 0);
    DstPtr = createEspVst128Ip(FuncBuilder, V2TailL, DstPtr);
    // esp.src.q.m(sar, v_last_loaded, v1_next) = (V1TailL, V0New2L).
    Value *TailCombinedL = createEspSrcQM(FuncBuilder, Sar2, V1TailL, V0New2L);
    DstPtr = createEspVst128Ip(FuncBuilder, TailCombinedL, DstPtr);

    if (Remainder > 32) {
      Value *SrcPtrAfter = FuncBuilder.CreateGEP(
          FuncBuilder.getInt8Ty(), SrcArg3L, FuncBuilder.getInt32(-32),
          "src.ptr.after.chunk3x16");
      FuncBuilder.CreateMemCpy(DstPtr, Align(1), SrcPtrAfter, Align(1),
                               FuncBuilder.getInt32(Remainder - 32));
    }
    FuncBuilder.CreateRetVoid();
    M->eraseFromParent();
    return true;
  }

  BasicBlock *EntryBB = nullptr, *ForBodyBB = nullptr,
             *ForCondCleanupBB = nullptr;
  createLoopBlocks(MemCpyFunc, EntryBB, ForBodyBB, ForCondCleanupBB);

  FuncBuilder.SetInsertPoint(EntryBB);
  Value *SrcPtr = SrcArg;
  auto [VecData1, Ptr1, Sar1] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
  SrcPtr = Ptr1;
  auto [VecData2, Ptr2, Sar2] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
  SrcPtr = Ptr2;
  FuncBuilder.CreateBr(ForBodyBB);

  FuncBuilder.SetInsertPoint(ForBodyBB);
  // Each loop trip: 3x vld/vst of 16B => 48B chunk (name chunk3x16, not array
  // index).
  PHINode *Chunk3x16LoopCounter = FuncBuilder.CreatePHI(
      FuncBuilder.getInt32Ty(), 2, "chunk3x16.loop.counter");
  Chunk3x16LoopCounter->addIncoming(FuncBuilder.getInt32(0), EntryBB);

  PHINode *SrcPtrInChunk3x16Loop =
      FuncBuilder.CreatePHI(PtrTy, 2, "src.ptr.in.chunk3x16.loop");
  SrcPtrInChunk3x16Loop->addIncoming(SrcPtr, EntryBB);

  PHINode *DstPtrInChunk3x16Loop =
      FuncBuilder.CreatePHI(PtrTy, 2, "dst.ptr.in.chunk3x16.loop");
  DstPtrInChunk3x16Loop->addIncoming(DstPtr, EntryBB);

  Type *V16I8 = VectorType::get(FuncBuilder.getInt8Ty(), 16, false);
  PHINode *V0Chunk3x16 = FuncBuilder.CreatePHI(V16I8, 2, "v0.chunk3x16");
  V0Chunk3x16->addIncoming(VecData1, EntryBB);
  PHINode *V1Chunk3x16 = FuncBuilder.CreatePHI(V16I8, 2, "v1.chunk3x16");
  V1Chunk3x16->addIncoming(VecData2, EntryBB);

  Value *Chunk3x16LoopCounterIncremented =
      FuncBuilder.CreateAdd(Chunk3x16LoopCounter, FuncBuilder.getInt32(1),
                            "chunk3x16.loop.counter.incremented", true, true);
  Chunk3x16LoopCounter->addIncoming(Chunk3x16LoopCounterIncremented, ForBodyBB);

  // Two PHIs only (v0, v1); third srcqldip uses (V2N2B4, V1NB4); back edge
  // v0<-V2N2B4, v1<-V0N2B4 so cleanup can use loop-body SSA only (no esp.orq).
  Value *DstPtrChunk3x16 = DstPtrInChunk3x16Loop;
  auto [V2NB4, V1NB4, SrcArg1B] = createEspSrcQLdIp(
      FuncBuilder, Sar2, V1Chunk3x16, V0Chunk3x16, SrcPtrInChunk3x16Loop, 16);
  DstPtrChunk3x16 = createEspVst128Ip(FuncBuilder, V2NB4, DstPtrChunk3x16);
  auto [V0NB4, V2N2B4, SrcArg2B] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V1NB4, V1Chunk3x16, SrcArg1B, 16);
  DstPtrChunk3x16 = createEspVst128Ip(FuncBuilder, V0NB4, DstPtrChunk3x16);
  auto [V1N2B4, V0N2B4, SrcArg3B] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V2N2B4, V1NB4, SrcArg2B, 16);
  DstPtrChunk3x16 = createEspVst128Ip(FuncBuilder, V1N2B4, DstPtrChunk3x16);

  SrcPtrInChunk3x16Loop->addIncoming(SrcArg3B, ForBodyBB);
  DstPtrInChunk3x16Loop->addIncoming(DstPtrChunk3x16, ForBodyBB);
  V0Chunk3x16->addIncoming(V2N2B4, ForBodyBB);
  V1Chunk3x16->addIncoming(V0N2B4, ForBodyBB);

  Value *LoopCompleted = FuncBuilder.CreateICmpEQ(
      Chunk3x16LoopCounterIncremented, FuncBuilder.getInt32(Quotient),
      "loop.completed");
  FuncBuilder.CreateCondBr(LoopCompleted, ForCondCleanupBB, ForBodyBB);

  FuncBuilder.SetInsertPoint(ForCondCleanupBB);
  // Use loop-body SSA only (V0N2B4, V2N2B4, SrcArg3B, DstPtrChunk3x16) so
  // cleanup does not use vector/ptr PHIs; reduces back-edge copies (esp.orq).
  // Cleanup: (qy,qw)=(V0N2B4,V2N2B4); then src.q.m(V1Tail,V0N2B4).
  auto [V2Tail, V1Tail, SrcArgTail] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V0N2B4, V2N2B4, SrcArg3B, 0);
  Value *DstPtrTail = createEspVst128Ip(FuncBuilder, V2Tail, DstPtrChunk3x16);
  Value *TailCombined = createEspSrcQM(FuncBuilder, Sar2, V1Tail, V0N2B4);
  DstPtrTail = createEspVst128Ip(FuncBuilder, TailCombined, DstPtrTail);

  if (Remainder > 32) {
    Value *SrcPtrAfterChunk3x16Loop = FuncBuilder.CreateGEP(
        FuncBuilder.getInt8Ty(), SrcArg3B, FuncBuilder.getInt32(-32),
        "src.ptr.after.chunk3x16.loop");
    FuncBuilder.CreateMemCpy(DstPtrTail, Align(1), SrcPtrAfterChunk3x16Loop,
                             Align(1), FuncBuilder.getInt32(Remainder - 32));
  }

  FuncBuilder.CreateRetVoid();

  M->eraseFromParent();
  return true;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstMod48From16To31(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t Quotient,
    uint64_t Remainder) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  static int FuncCounter = 0;
  std::string FuncName = "esp32p4MemCpySrcunalignedDst16mod48From16to31." +
                         std::to_string(FuncCounter++);

  Function *MemCpyFunc =
      createMemCpyHelperFunctionPtrNoSize(Builder, FuncName, Dst, Src, false);
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");

  // Quotient==1: generate linear (fully unrolled) IR to avoid PHI nodes and
  // eliminate esp.orq copies that back up exit values for cleanup.
  if (Quotient == 1) {
    BasicBlock *EntryBB =
        BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
    IRBuilder<> FuncBuilder(EntryBB);
    Value *SrcPtr = SrcArg;
    Value *DstPtr = DstArg;

    auto [VecData1, Ptr1, Sar1] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
    SrcPtr = Ptr1;
    auto [VecData2, Ptr2, Sar2] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
    SrcPtr = Ptr2;

    // Single iteration: three src.q.ld.ip + vst.128.ip, no loop/PHI.
    auto [V2NB3, V1NB3, SrcArg1B3] =
        createEspSrcQLdIp(FuncBuilder, Sar2, VecData2, VecData1, SrcPtr, 16);
    DstPtr = createEspVst128Ip(FuncBuilder, V2NB3, DstPtr);
    auto [V0NB3, V2N2B3, SrcArg2B3] =
        createEspSrcQLdIp(FuncBuilder, Sar2, V1NB3, VecData2, SrcArg1B3, 16);
    DstPtr = createEspVst128Ip(FuncBuilder, V0NB3, DstPtr);
    auto [V1N2B3, V0N2B3, SrcArg3B3] =
        createEspSrcQLdIp(FuncBuilder, Sar2, V2N2B3, V1NB3, SrcArg2B3, 16);
    DstPtr = createEspVst128Ip(FuncBuilder, V1N2B3, DstPtr);

    // Tail: esp.src.q.m(sar, qy, qw). arg1=qy from index-2 src.q.ld.ip.m res
    // (V0N2B3), arg2=qw from index-1 src.q.ld.ip.m res (V2N2B3).
    Value *TailBlock = createEspSrcQM(FuncBuilder, Sar2, V0N2B3, V2N2B3);
    Value *DstPtrTail3 = createEspVst128Ip(FuncBuilder, TailBlock, DstPtr);
    Value *SrcPtrAfterChunk3x16Loop = FuncBuilder.CreateGEP(
        FuncBuilder.getInt8Ty(), SrcArg3B3, FuncBuilder.getInt32(-32),
        "src.ptr.after.chunk3x16.loop");
    FuncBuilder.CreateMemCpy(DstPtrTail3, Align(1), SrcPtrAfterChunk3x16Loop,
                             Align(1), FuncBuilder.getInt32(Remainder - 16));
    FuncBuilder.CreateRetVoid();

    M->eraseFromParent();
    return true;
  }

  BasicBlock *EntryBB = nullptr, *ForBodyBB = nullptr,
             *ForCondCleanupBB = nullptr;
  createLoopBlocks(MemCpyFunc, EntryBB, ForBodyBB, ForCondCleanupBB);

  IRBuilder<> FuncBuilder(EntryBB);
  Type *PtrTy = FuncBuilder.getPtrTy();
  Value *SrcPtr = SrcArg;
  Value *DstPtr = DstArg;

  auto [VecData1, Ptr1, Sar1] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
  SrcPtr = Ptr1;
  auto [VecData2, Ptr2, Sar2] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
  SrcPtr = Ptr2;
  FuncBuilder.CreateBr(ForBodyBB);

  FuncBuilder.SetInsertPoint(ForBodyBB);
  PHINode *Chunk3x16LoopCounter = FuncBuilder.CreatePHI(
      FuncBuilder.getInt32Ty(), 2, "chunk3x16.loop.counter");
  Chunk3x16LoopCounter->addIncoming(FuncBuilder.getInt32(0), EntryBB);

  PHINode *SrcPtrInChunk3x16Loop =
      FuncBuilder.CreatePHI(PtrTy, 2, "src.ptr.in.chunk3x16.loop");
  SrcPtrInChunk3x16Loop->addIncoming(SrcPtr, EntryBB);

  PHINode *DstPtrInChunk3x16Loop =
      FuncBuilder.CreatePHI(PtrTy, 2, "dst.ptr.in.chunk3x16.loop");
  DstPtrInChunk3x16Loop->addIncoming(DstPtr, EntryBB);

  Type *V16I8_3 = VectorType::get(FuncBuilder.getInt8Ty(), 16, false);
  PHINode *V0B3 = FuncBuilder.CreatePHI(V16I8_3, 2, "v0.chunk3x16.3");
  V0B3->addIncoming(VecData1, EntryBB);
  PHINode *V1B3 = FuncBuilder.CreatePHI(V16I8_3, 2, "v1.chunk3x16.3");
  V1B3->addIncoming(VecData2, EntryBB);

  Value *Chunk3x16LoopCounterIncremented =
      FuncBuilder.CreateAdd(Chunk3x16LoopCounter, FuncBuilder.getInt32(1),
                            "chunk3x16.loop.counter.incremented", true, true);
  Chunk3x16LoopCounter->addIncoming(Chunk3x16LoopCounterIncremented, ForBodyBB);

  Value *DstPtrB3 = DstPtrInChunk3x16Loop;
  auto [V2NB3, V1NB3, SrcArg1B3] = createEspSrcQLdIp(
      FuncBuilder, Sar2, V1B3, V0B3, SrcPtrInChunk3x16Loop, 16);
  DstPtrB3 = createEspVst128Ip(FuncBuilder, V2NB3, DstPtrB3);
  auto [V0NB3, V2N2B3, SrcArg2B3] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V1NB3, V1B3, SrcArg1B3, 16);
  DstPtrB3 = createEspVst128Ip(FuncBuilder, V0NB3, DstPtrB3);
  auto [V1N2B3, V0N2B3, SrcArg3B3] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V2N2B3, V1NB3, SrcArg2B3, 16);
  DstPtrB3 = createEspVst128Ip(FuncBuilder, V1N2B3, DstPtrB3);

  SrcPtrInChunk3x16Loop->addIncoming(SrcArg3B3, ForBodyBB);
  DstPtrInChunk3x16Loop->addIncoming(DstPtrB3, ForBodyBB);
  // PHI back-edge: feed last two loads' index-1 results (second srcqldip -> v0,
  // third srcqldip -> v1).
  V0B3->addIncoming(V2N2B3, ForBodyBB);
  V1B3->addIncoming(V0N2B3, ForBodyBB);

  Value *LoopCompleted = FuncBuilder.CreateICmpEQ(
      Chunk3x16LoopCounterIncremented, FuncBuilder.getInt32(Quotient),
      "loop.completed");
  FuncBuilder.CreateCondBr(LoopCompleted, ForCondCleanupBB, ForBodyBB);

  FuncBuilder.SetInsertPoint(ForCondCleanupBB);
  // Use loop-body SSA values (V0N2B3, V2N2B3, DstPtrB3, SrcArg3B3) in cleanup
  // so cleanup does not use vector PHIs; reduces pressure for back-edge copies.
  Value *TailBlock = createEspSrcQM(FuncBuilder, Sar2, V0N2B3, V2N2B3);
  Value *DstPtrTail3 = createEspVst128Ip(FuncBuilder, TailBlock, DstPtrB3);
  Value *SrcPtrAfterChunk3x16Loop = FuncBuilder.CreateGEP(
      FuncBuilder.getInt8Ty(), SrcArg3B3, FuncBuilder.getInt32(-32),
      "src.ptr.after.chunk3x16.loop");

  FuncBuilder.CreateMemCpy(DstPtrTail3, Align(1), SrcPtrAfterChunk3x16Loop,
                           Align(1), FuncBuilder.getInt32(Remainder - 16));

  FuncBuilder.CreateRetVoid();

  M->eraseFromParent();
  return true;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstMod48From1To15(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t Quotient,
    uint64_t Remainder) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  static int FuncCounter = 0;
  std::string FuncName = "esp32p4_memcpy_srcunaligned_dst16mod48_1to15Index" +
                         std::to_string(FuncCounter++);

  Function *MemCpyFunc =
      createMemCpyHelperFunctionPtrNoSize(Builder, FuncName, Dst, Src);
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");

  BasicBlock *EntryBB = nullptr, *ForBodyBB = nullptr,
             *ForCondCleanupBB = nullptr;
  createLoopBlocks(MemCpyFunc, EntryBB, ForBodyBB, ForCondCleanupBB);

  IRBuilder<> FuncBuilder(EntryBB);
  Type *PtrTy = FuncBuilder.getPtrTy();
  Value *SrcPtr = SrcArg;
  Value *DstPtr = DstArg;

  auto [VecData1, Ptr1, Sar1] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
  SrcPtr = Ptr1;
  auto [VecData2, Ptr2, Sar2] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
  SrcPtr = Ptr2;
  FuncBuilder.CreateBr(ForBodyBB);

  FuncBuilder.SetInsertPoint(ForBodyBB);
  PHINode *Chunk3x16LoopCounter = FuncBuilder.CreatePHI(
      FuncBuilder.getInt32Ty(), 2, "chunk3x16.loop.counter");
  Chunk3x16LoopCounter->addIncoming(FuncBuilder.getInt32(0), EntryBB);

  PHINode *SrcPtrInChunk3x16Loop =
      FuncBuilder.CreatePHI(PtrTy, 2, "src.ptr.in.chunk3x16.loop");
  SrcPtrInChunk3x16Loop->addIncoming(SrcPtr, EntryBB);

  PHINode *DstPtrInChunk3x16Loop =
      FuncBuilder.CreatePHI(PtrTy, 2, "dst.ptr.in.chunk3x16.loop");
  DstPtrInChunk3x16Loop->addIncoming(DstPtr, EntryBB);

  Type *V16I8_4 = VectorType::get(FuncBuilder.getInt8Ty(), 16, false);
  PHINode *V0B4 = FuncBuilder.CreatePHI(V16I8_4, 2, "v0.chunk3x16.4");
  V0B4->addIncoming(VecData1, EntryBB);
  PHINode *V1B4 = FuncBuilder.CreatePHI(V16I8_4, 2, "v1.chunk3x16.4");
  V1B4->addIncoming(VecData2, EntryBB);
  PHINode *V2B4 = FuncBuilder.CreatePHI(V16I8_4, 2, "v2.chunk3x16.4");
  V2B4->addIncoming(PoisonValue::get(V16I8_4), EntryBB);

  Value *Chunk3x16LoopCounterIncremented =
      FuncBuilder.CreateAdd(Chunk3x16LoopCounter, FuncBuilder.getInt32(1),
                            "chunk3x16.loop.counter.incremented", true, true);
  Chunk3x16LoopCounter->addIncoming(Chunk3x16LoopCounterIncremented, ForBodyBB);

  Value *DstPtrB4 = DstPtrInChunk3x16Loop;
  auto [V2NB4, V1NB4, SrcArg1B4] = createEspSrcQLdIp(
      FuncBuilder, Sar2, V0B4, V1B4, SrcPtrInChunk3x16Loop, 16);
  DstPtrB4 = createEspVst128Ip(FuncBuilder, V0B4, DstPtrB4);
  auto [V0NB4, V2N2B4, SrcArg2B4] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V1NB4, V2NB4, SrcArg1B4, 16);
  DstPtrB4 = createEspVst128Ip(FuncBuilder, V1NB4, DstPtrB4);
  auto [V1N2B4, V0N2B4, SrcArg3B4] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V2N2B4, V0NB4, SrcArg2B4, 16);
  DstPtrB4 = createEspVst128Ip(FuncBuilder, V2N2B4, DstPtrB4);

  SrcPtrInChunk3x16Loop->addIncoming(SrcArg3B4, ForBodyBB);
  DstPtrInChunk3x16Loop->addIncoming(DstPtrB4, ForBodyBB);
  V0B4->addIncoming(V0N2B4, ForBodyBB);
  V1B4->addIncoming(V1N2B4, ForBodyBB);
  V2B4->addIncoming(V2N2B4, ForBodyBB);

  Value *LoopCompleted = FuncBuilder.CreateICmpEQ(
      Chunk3x16LoopCounterIncremented, FuncBuilder.getInt32(Quotient),
      "loop.completed");
  FuncBuilder.CreateCondBr(LoopCompleted, ForCondCleanupBB, ForBodyBB);

  FuncBuilder.SetInsertPoint(ForCondCleanupBB);
  // Use loop-body SSA (SrcArg3B4, DstPtrB4) in cleanup so cleanup does not use
  // pointer PHIs; reduces pressure for back-edge copies (esp.orq).
  Value *SrcPtrAfterChunk3x16Loop = FuncBuilder.CreateGEP(
      FuncBuilder.getInt8Ty(), SrcArg3B4, FuncBuilder.getInt32(-32),
      "src.ptr.after.chunk3x16.loop");

  FuncBuilder.CreateMemCpy(DstPtrB4, Align(1), SrcPtrAfterChunk3x16Loop,
                           Align(1), FuncBuilder.getInt32(Remainder - 32));

  FuncBuilder.CreateRetVoid();

  M->eraseFromParent();
  return true;
}
// src unaligned, dst 16-byte aligned, size is divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16Const16(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  uint64_t Quotient = Len / 48;
  uint64_t Remainder = Len % 48;
  if (Len < 16) {
    return false; // not process, use memcpy
  }
  if (Quotient == 0) {
    if (Remainder >= 1 && Remainder <= 15) {
      return true;
    }

    if (Remainder >= 16 && Remainder <= 31) {

      IRBuilder<> Builder(M);
      Value *Src = M->getSource();
      Value *Dst = M->getDest();

      static int FuncCounter = 0;
      std::string FuncName = "esp32p4MemCpySrcunalignedDst16From16to31Index" +
                             std::to_string(FuncCounter++);

      Function *MemCpyFunc =
          createMemCpyHelperFunctionPtrNoSize(Builder, FuncName, Dst, Src);

      Value *DstArg = MemCpyFunc->arg_begin();
      DstArg->setName("dst");
      Value *SrcArg = MemCpyFunc->arg_begin() + 1;
      SrcArg->setName("src");

      BasicBlock *EntryBB =
          BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
      IRBuilder<> FuncBuilder(EntryBB);
      FuncBuilder.SetInsertPoint(EntryBB);
      // Capture both load vectors and combine for unaligned 16-byte block.
      // src.q.m(sar, qy, qw): shift concatenated (qy||qw); use (Vec2, Vec1) to
      // match 16-byte tail convention.
      auto [Vec1, Ptr1, Sar1] = createEspLd128UsarIp(FuncBuilder, SrcArg);
      auto [Vec2, Ptr2, Sar2] = createEspLd128UsarIp(FuncBuilder, Ptr1);
      Value *Combined = createEspSrcQM(FuncBuilder, Sar2, Vec2, Vec1);
      Value *DstPtr = createEspVst128Ip(FuncBuilder, Combined, DstArg);

      if (Remainder > 16)
        FuncBuilder.CreateMemCpy(DstPtr, Align(16), Ptr2, Align(1),
                                 FuncBuilder.getInt32(Remainder - 16));

      FuncBuilder.CreateRetVoid();
      M->eraseFromParent();
      return true;
    } else if (Remainder >= 32 && Remainder <= 47) {
      IRBuilder<> Builder(M);
      Value *Src = M->getSource();
      Value *Dst = M->getDest();

      std::string FuncName = "esp32p4MemCpySrcunalignedDst16From32To47";
      if (Function *ExistingFunc = M->getModule()->getFunction(FuncName)) {
        Builder.CreateCall(ExistingFunc, {Dst, Src});
        M->eraseFromParent();
        return true;
      }

      Function *MemCpyFunc =
          createMemCpyHelperFunctionPtrNoSize(Builder, FuncName, Dst, Src);
      Value *DstArg = MemCpyFunc->arg_begin();
      DstArg->setName("dst");
      Value *SrcArg = MemCpyFunc->arg_begin() + 1;
      SrcArg->setName("src");

      BasicBlock *EntryBB =
          BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
      IRBuilder<> FuncBuilder(EntryBB);
      FuncBuilder.SetInsertPoint(EntryBB);
      auto [VecData1R, Ptr1R, Sar1R] =
          createEspLd128UsarIp(FuncBuilder, SrcArg);
      Value *SrcPtrR = Ptr1R;
      auto [VecData2R, Ptr2R, Sar2R] =
          createEspLd128UsarIp(FuncBuilder, SrcPtrR);
      SrcPtrR = Ptr2R;

      // src.q.ld.ip(sar, qy, qw): same qy/qw order as 16-byte case (second,
      // first).
      auto [V2R, V1R, SrcArgR] = createEspSrcQLdIp(
          FuncBuilder, Sar2R, VecData2R, VecData1R, SrcPtrR, 0);
      // First 16 bytes: store shifted result (QwNew), not raw first load.
      Value *DstPtrR = createEspVst128Ip(FuncBuilder, V2R, DstArg);
      // Second 16 bytes: combine VecData2R and QuNew (same qy/qw order as
      // 16-byte case).
      Value *SecondBlock = createEspSrcQM(FuncBuilder, Sar2R, V1R, VecData2R);
      DstPtrR = createEspVst128Ip(FuncBuilder, SecondBlock, DstPtrR);

      FuncBuilder.CreateMemCpy(DstPtrR, Align(16), SrcArgR, Align(1),
                               FuncBuilder.getInt32(Remainder - 32));

      FuncBuilder.CreateRetVoid();
      M->eraseFromParent();
      return true;
    }
    return false; // not process, use memcpy
  } else {
    if (Remainder == 0) {
      return processSrcUnalignDst16ConstDiv48(M, BBI, Quotient);
    } else if (Remainder >= 32 && Remainder <= 47) {
      return processSrcUnalignDst16ConstMod48From32To47(M, BBI, Quotient,
                                                        Remainder);
    } else if (Remainder >= 16 && Remainder <= 31) {
      return processSrcUnalignDst16ConstMod48From16To31(M, BBI, Quotient,
                                                        Remainder);
    } else if (Remainder >= 1 && Remainder <= 15) {
      return processSrcUnalignDst16ConstMod48From1To15(M, BBI, Quotient,
                                                       Remainder);
    }
  }
  return false;
}

// src unaligned, dst 16-byte aligned, size is divisible by 8
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16Const8(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  if (Len == 8)
    return false;
  uint64_t mainSize = Len - 8;
  uint64_t srcAlign = M->getSourceAlign()->value();
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();
  Builder.CreateMemCpy(Dst, Align(16), Src, Align(srcAlign), mainSize);

  Value *NewSrc =
      Builder.CreateGEP(Builder.getInt8Ty(), Src, Builder.getInt64(mainSize));
  Value *NewDst =
      Builder.CreateGEP(Builder.getInt8Ty(), Dst, Builder.getInt64(mainSize));

  Builder.CreateMemCpy(NewDst, Align(16), NewSrc, Align(srcAlign), 8);

  M->eraseFromParent();
  return true;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16Common(
    MemCpyInst *M, BasicBlock::iterator &BBI, bool isInline) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();
  Value *Size = M->getLength();

  std::string FuncName = "esp32p4MemCpySrcunalignedDst16Var";

  if (useExistingHelperFunction(M, Builder, FuncName, Dst, Src, Size)) {
    return true;
  }
  Function *MemCpyFunc = createMemCpyHelperFunctionPtr(Builder, FuncName, Dst,
                                                       Src, Size, isInline);
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *DstArgOriginal = DstArg;
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");
  Value *SrcArgOriginal = SrcArg;
  Value *SizeArg = MemCpyFunc->arg_begin() + 2;
  SizeArg->setName("size");

  BasicBlock *EntryBB =
      BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
  IRBuilder<> FuncBuilder(EntryBB);
  Type *PtrTy = FuncBuilder.getPtrTy();
  Type *V16I8 = VectorType::get(FuncBuilder.getInt8Ty(), 16, false);

  BasicBlock *ProcessMainLoopBB =
      BasicBlock::Create(M->getContext(), "process.main.loop", MemCpyFunc);
  BasicBlock *HandleRemainderBB =
      BasicBlock::Create(M->getContext(), "handle.Remainder", MemCpyFunc);
  BasicBlock *MainLoopBodyBB =
      BasicBlock::Create(M->getContext(), "main.loop.body", MemCpyFunc);
  BasicBlock *Process32ByteTailBB =
      BasicBlock::Create(M->getContext(), "process.32byte.tail", MemCpyFunc);
  BasicBlock *Check16ByteTailBB =
      BasicBlock::Create(M->getContext(), "check.16byte.tail", MemCpyFunc);
  BasicBlock *Process16ByteTailBB =
      BasicBlock::Create(M->getContext(), "process.16byte.tail", MemCpyFunc);
  BasicBlock *SkipTailProcessingBB =
      BasicBlock::Create(M->getContext(), "skip.tail.processing", MemCpyFunc);
  BasicBlock *FinalCleanupBB =
      BasicBlock::Create(M->getContext(), "final.cleanup", MemCpyFunc);
  BasicBlock *CallSmallSizeCleanupBB = BasicBlock::Create(
      M->getContext(), "call.small.size.cleanup", MemCpyFunc);
  BasicBlock *ReturnBB =
      BasicBlock::Create(M->getContext(), "return", MemCpyFunc);

  Value *SizeIsSmall = FuncBuilder.CreateICmpULT(
      SizeArg, FuncBuilder.getInt32(16), "size.is.small");
  FuncBuilder.CreateCondBr(SizeIsSmall, FinalCleanupBB, ProcessMainLoopBB);

  FuncBuilder.SetInsertPoint(ProcessMainLoopBB);
  Value *Blocks48Count = FuncBuilder.CreateUDiv(
      SizeArg, FuncBuilder.getInt32(48), "blocks.48.count");
  Value *Blocks48TotalBytes =
      FuncBuilder.CreateMul(Blocks48Count, FuncBuilder.getInt32(48));
  Value *RemainderAfter48Blocks = FuncBuilder.CreateSub(
      SizeArg, Blocks48TotalBytes, "Remainder.after.48blocks");

  Value *SrcPtr = SrcArg;
  Value *DstPtr = DstArg;
  auto [VecData1, Ptr1, Sar1] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
  SrcPtr = Ptr1;
  auto [VecData2, Ptr2, Sar2] = createEspLd128UsarIp(FuncBuilder, SrcPtr);
  SrcPtr = Ptr2;

  Value *Has48ByteBlocks = FuncBuilder.CreateICmpULT(
      SizeArg, FuncBuilder.getInt32(48), "no.48byte.blocks");
  FuncBuilder.CreateCondBr(Has48ByteBlocks, HandleRemainderBB, MainLoopBodyBB);

  FuncBuilder.SetInsertPoint(HandleRemainderBB);
  PHINode *SrcPtrAfterMainLoop =
      FuncBuilder.CreatePHI(PtrTy, 2, "src.ptr.after.main.loop");
  SrcPtrAfterMainLoop->addIncoming(SrcPtr, ProcessMainLoopBB);
  PHINode *DstPtrAfterMainLoop =
      FuncBuilder.CreatePHI(PtrTy, 2, "dst.ptr.after.main.loop");
  DstPtrAfterMainLoop->addIncoming(DstPtr, ProcessMainLoopBB);
  PHINode *V0AfterMainLoop =
      FuncBuilder.CreatePHI(V16I8, 2, "v0.after.main.loop");
  V0AfterMainLoop->addIncoming(VecData1, ProcessMainLoopBB);
  PHINode *V1AfterMainLoop =
      FuncBuilder.CreatePHI(V16I8, 2, "v1.after.main.loop");
  V1AfterMainLoop->addIncoming(VecData2, ProcessMainLoopBB);

  Value *RemainderHas32Bytes = FuncBuilder.CreateICmpULT(
      RemainderAfter48Blocks, FuncBuilder.getInt32(32),
      "Remainder.has.no.32bytes");
  FuncBuilder.CreateCondBr(RemainderHas32Bytes, Check16ByteTailBB,
                           Process32ByteTailBB);

  FuncBuilder.SetInsertPoint(MainLoopBodyBB);
  PHINode *LoopIndex =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "loop.index");
  LoopIndex->addIncoming(FuncBuilder.getInt32(0), ProcessMainLoopBB);

  PHINode *SrcPtrInLoop = FuncBuilder.CreatePHI(PtrTy, 2, "src.ptr.in.loop");
  SrcPtrInLoop->addIncoming(SrcPtr, ProcessMainLoopBB);
  PHINode *DstPtrInLoop = FuncBuilder.CreatePHI(PtrTy, 2, "dst.ptr.in.loop");
  DstPtrInLoop->addIncoming(DstPtr, ProcessMainLoopBB);
  PHINode *V0 = FuncBuilder.CreatePHI(V16I8, 2, "v0");
  V0->addIncoming(VecData1, ProcessMainLoopBB);
  PHINode *V1 = FuncBuilder.CreatePHI(V16I8, 2, "v1");
  V1->addIncoming(VecData2, ProcessMainLoopBB);

  Value *DstPtrVar = DstPtrInLoop;
  auto [V2New, V1New, SrcArg1] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V1, V0, SrcPtrInLoop, 16);
  DstPtrVar = createEspVst128Ip(FuncBuilder, V2New, DstPtrVar);
  auto [V0New, V2New2, SrcArg2] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V1New, V1, SrcArg1, 16);
  DstPtrVar = createEspVst128Ip(FuncBuilder, V0New, DstPtrVar);
  auto [V1New2, V0New2, SrcArg3] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V2New2, V1New, SrcArg2, 16);
  DstPtrVar = createEspVst128Ip(FuncBuilder, V1New2, DstPtrVar);

  SrcPtrInLoop->addIncoming(SrcArg3, MainLoopBodyBB);
  DstPtrInLoop->addIncoming(DstPtrVar, MainLoopBodyBB);
  V0->addIncoming(V2New2, MainLoopBodyBB);
  V1->addIncoming(V0New2, MainLoopBodyBB);

  Value *LoopIndexIncremented = FuncBuilder.CreateAdd(
      LoopIndex, FuncBuilder.getInt32(1), "loop.index.incremented", true, true);
  LoopIndex->addIncoming(LoopIndexIncremented, MainLoopBodyBB);
  Value *LoopCompleted = FuncBuilder.CreateICmpEQ(
      LoopIndexIncremented, Blocks48Count, "loop.completed");
  FuncBuilder.CreateCondBr(LoopCompleted, HandleRemainderBB, MainLoopBodyBB);

  SrcPtrAfterMainLoop->addIncoming(SrcArg3, MainLoopBodyBB);
  DstPtrAfterMainLoop->addIncoming(DstPtrVar, MainLoopBodyBB);
  V0AfterMainLoop->addIncoming(V2New2, MainLoopBodyBB);
  V1AfterMainLoop->addIncoming(V0New2, MainLoopBodyBB);

  FuncBuilder.SetInsertPoint(Process32ByteTailBB);
  auto [V2Tail, V1Tail, SrcAfter32ByteProcessing] =
      createEspSrcQLdIp(FuncBuilder, Sar2, V1AfterMainLoop, V0AfterMainLoop,
                        SrcPtrAfterMainLoop, 0);
  Value *DstAfter32 =
      createEspVst128Ip(FuncBuilder, V2Tail, DstPtrAfterMainLoop);
  Value *Second32ByteBlock =
      createEspSrcQM(FuncBuilder, Sar2, V1Tail, V1AfterMainLoop);
  Value *DstAfter32ByteProcessing =
      createEspVst128Ip(FuncBuilder, Second32ByteBlock, DstAfter32);
  Value *RemainderAfter32ByteProcessing =
      FuncBuilder.CreateAdd(RemainderAfter48Blocks, FuncBuilder.getInt32(-32),
                            "Remainder.after.32byte.processing", false, true);
  FuncBuilder.CreateBr(FinalCleanupBB);

  FuncBuilder.SetInsertPoint(Check16ByteTailBB);
  Value *RemainderHas16Bytes = FuncBuilder.CreateICmpULT(
      RemainderAfter48Blocks, FuncBuilder.getInt32(16),
      "Remainder.has.no.16bytes");
  FuncBuilder.CreateCondBr(RemainderHas16Bytes, SkipTailProcessingBB,
                           Process16ByteTailBB);

  FuncBuilder.SetInsertPoint(Process16ByteTailBB);
  Value *Combined16ByteBlock =
      createEspSrcQM(FuncBuilder, Sar2, V1AfterMainLoop, V0AfterMainLoop);
  Value *DstAfter16ByteProcessing =
      createEspVst128Ip(FuncBuilder, Combined16ByteBlock, DstPtrAfterMainLoop);
  Value *SrcAfter16ByteProcessing = FuncBuilder.CreateGEP(
      FuncBuilder.getInt8Ty(), SrcPtrAfterMainLoop, FuncBuilder.getInt32(-16),
      "src.after.16byte.processing");
  Value *RemainderAfter16ByteProcessing =
      FuncBuilder.CreateAdd(RemainderAfter48Blocks, FuncBuilder.getInt32(-16),
                            "Remainder.after.16byte.processing", false, true);
  FuncBuilder.CreateBr(FinalCleanupBB);

  FuncBuilder.SetInsertPoint(SkipTailProcessingBB);
  PHINode *SrcForNoTailProcessing =
      FuncBuilder.CreatePHI(PtrTy, 1, "src.for.no.tail.processing");
  SrcForNoTailProcessing->addIncoming(SrcPtrAfterMainLoop, Check16ByteTailBB);
  PHINode *DstForNoTailProcessing =
      FuncBuilder.CreatePHI(PtrTy, 1, "dst.for.no.tail.processing");
  DstForNoTailProcessing->addIncoming(DstPtrAfterMainLoop, Check16ByteTailBB);
  PHINode *RemainderForNoTailProcessing = FuncBuilder.CreatePHI(
      FuncBuilder.getInt32Ty(), 1, "Remainder.for.no.tail.processing");
  RemainderForNoTailProcessing->addIncoming(RemainderAfter48Blocks,
                                            Check16ByteTailBB);
  Value *SrcAdjustedForUnaligned = FuncBuilder.CreateGEP(
      FuncBuilder.getInt8Ty(), SrcForNoTailProcessing,
      FuncBuilder.getInt32(-32), "src.adjusted.for.unaligned");
  FuncBuilder.CreateBr(FinalCleanupBB);

  FuncBuilder.SetInsertPoint(FinalCleanupBB);
  PHINode *SrcForFinalProcessing =
      FuncBuilder.CreatePHI(PtrTy, 4, "src.for.final.processing");
  SrcForFinalProcessing->addIncoming(SrcAfter32ByteProcessing,
                                     Process32ByteTailBB);
  SrcForFinalProcessing->addIncoming(SrcAfter16ByteProcessing,
                                     Process16ByteTailBB);
  SrcForFinalProcessing->addIncoming(SrcArgOriginal, EntryBB);
  SrcForFinalProcessing->addIncoming(SrcAdjustedForUnaligned,
                                     SkipTailProcessingBB);
  PHINode *DstForFinalProcessing =
      FuncBuilder.CreatePHI(PtrTy, 4, "dst.for.final.processing");
  DstForFinalProcessing->addIncoming(DstAfter32ByteProcessing,
                                     Process32ByteTailBB);
  DstForFinalProcessing->addIncoming(DstAfter16ByteProcessing,
                                     Process16ByteTailBB);
  DstForFinalProcessing->addIncoming(DstArgOriginal, EntryBB);
  DstForFinalProcessing->addIncoming(DstForNoTailProcessing,
                                     SkipTailProcessingBB);
  PHINode *RemainderForFinalProcessing = FuncBuilder.CreatePHI(
      FuncBuilder.getInt32Ty(), 4, "Remainder.for.final.processing");
  RemainderForFinalProcessing->addIncoming(SizeArg, EntryBB);
  RemainderForFinalProcessing->addIncoming(RemainderAfter32ByteProcessing,
                                           Process32ByteTailBB);
  RemainderForFinalProcessing->addIncoming(RemainderAfter16ByteProcessing,
                                           Process16ByteTailBB);
  RemainderForFinalProcessing->addIncoming(RemainderForNoTailProcessing,
                                           SkipTailProcessingBB);

  Value *HasRemainingBytes = FuncBuilder.CreateICmpEQ(
      RemainderForFinalProcessing, FuncBuilder.getInt32(0),
      "has.no.remaining.bytes");
  FuncBuilder.CreateCondBr(HasRemainingBytes, ReturnBB, CallSmallSizeCleanupBB);

  FuncBuilder.SetInsertPoint(ReturnBB);
  FuncBuilder.CreateRetVoid();

  FuncBuilder.SetInsertPoint(CallSmallSizeCleanupBB);
  processMemCpyVarFrom1To15(
      FuncBuilder, "esp32p4MemCpySrcUnalignDst16From1To15Opt",
      DstForFinalProcessing, SrcForFinalProcessing, RemainderForFinalProcessing,
      true, MemCpyType::SrcUnalign_Dst16_Var);
  FuncBuilder.CreateBr(ReturnBB);

  if (verifyFunction(*MemCpyFunc, &errs()))
    report_fatal_error(
        "RISCVEsp32P4MemIntrin: created invalid helper function");

  M->eraseFromParent();
  return true;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16Var(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrcUnalignDst16Common(M, BBI, true);
}

// src unaligned, dst 8-byte aligned, size is divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst8Const16(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  std::string FuncName = "esp32p4MemCpySrcunalignDst8Const16";

  // Check if the function already exists in the current TheModule
  if (useExistingHelperFunction(M, Builder, FuncName, Dst, Src,
                                Builder.getInt32(Len))) {
    return true;
  }

  Function *MemCpyFunc = createMemCpyHelperFunctionPtr(
      Builder, FuncName, Dst, Src, Builder.getInt32(Len), true);
  // Create entry block
  BasicBlock *EntryBB =
      BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
  IRBuilder<> FuncBuilder(EntryBB);

  // Extract function arguments
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");
  Value *SizeArg = MemCpyFunc->arg_begin() + 2;
  SizeArg->setName("size");

  // Load and store the first 8 bytes as two i32 pairs (RV32-friendly, no i64).
  Value *Load0 =
      FuncBuilder.CreateAlignedLoad(FuncBuilder.getInt32Ty(), SrcArg, Align(1));
  FuncBuilder.CreateAlignedStore(Load0, DstArg, Align(1));
  Value *Src4 = FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), SrcArg,
                                      FuncBuilder.getInt32(4), "src.4");
  Value *Dst4 = FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), DstArg,
                                      FuncBuilder.getInt32(4), "dst.4");
  Value *Load1 =
      FuncBuilder.CreateAlignedLoad(FuncBuilder.getInt32Ty(), Src4, Align(1));
  FuncBuilder.CreateAlignedStore(Load1, Dst4, Align(1));

  // Calculate the remaining size after processing first 8 bytes
  Value *RemainingSizeAfterFirst8Bytes = FuncBuilder.CreateSub(
      SizeArg, FuncBuilder.getInt32(8), "remaining.size.after.first.8bytes");

  Value *DstOffsetBy8Bytes =
      FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), DstArg,
                            FuncBuilder.getInt32(8), "dst.offset.by.8bytes");
  Value *SrcOffsetBy8Bytes =
      FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), SrcArg,
                            FuncBuilder.getInt32(8), "src.offset.by.8bytes");

  FuncBuilder.CreateMemCpy(DstOffsetBy8Bytes, Align(16), SrcOffsetBy8Bytes,
                           Align(1), RemainingSizeAfterFirst8Bytes);

  FuncBuilder.CreateRetVoid();

  M->eraseFromParent();
  return true;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst8Const8(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  if (Len == 8)
    return false;
  uint64_t mainSize = Len - 8;
  uint64_t srcAlign = M->getSourceAlign()->value();
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();
  Builder.CreateMemCpy(Dst, Align(8), Src, Align(srcAlign), 8);

  Value *NewSrc =
      Builder.CreateGEP(Builder.getInt8Ty(), Src, Builder.getInt64(8));
  Value *NewDst =
      Builder.CreateGEP(Builder.getInt8Ty(), Dst, Builder.getInt64(8));

  Builder.CreateMemCpy(NewDst, Align(16), NewSrc, Align(srcAlign), mainSize);

  M->eraseFromParent();
  return true;
}

// src unaligned, dst 8-byte aligned, size is variable
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst8Var(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();
  Value *Size = M->getLength();

  std::string FuncName = "esp32p4MemCpySrcUnalignDst8Var";
  // Check if the function already exists in the current TheModule
  if (useExistingHelperFunction(M, Builder, FuncName, Dst, Src, Size)) {
    return true;
  }

  Function *MemCpyFunc =
      createMemCpyHelperFunctionPtr(Builder, FuncName, Dst, Src, Size, true);
  // Create entry block
  BasicBlock *EntryBB =
      BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
  IRBuilder<> FuncBuilder(EntryBB);

  // Extract function arguments
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");
  Value *SizeArg = MemCpyFunc->arg_begin() + 2;
  SizeArg->setName("size");

  // Create basic blocks
  BasicBlock *IfThenBB =
      BasicBlock::Create(M->getContext(), "if.then", MemCpyFunc);
  BasicBlock *IfEndBB =
      BasicBlock::Create(M->getContext(), "if.end", MemCpyFunc);
  BasicBlock *ReturnBB =
      BasicBlock::Create(M->getContext(), "return", MemCpyFunc);

  // entry block
  Value *Cmp = FuncBuilder.CreateICmpULT(SizeArg, FuncBuilder.getInt32(8));
  FuncBuilder.CreateCondBr(Cmp, IfThenBB, IfEndBB);

  // if.end block
  FuncBuilder.SetInsertPoint(IfThenBB);

  processMemCpyVarFrom1To7(FuncBuilder,
                           "esp32p4MemCpySrcUnalignDst8From1To7Opt", DstArg,
                           SrcArg, SizeArg, true);

  FuncBuilder.CreateBr(ReturnBB);

  // if.then9 block
  FuncBuilder.SetInsertPoint(IfEndBB);

  // Load and store the first 8 bytes as two i32 pairs (RV32-friendly, no i64).
  Value *Load0 =
      FuncBuilder.CreateAlignedLoad(FuncBuilder.getInt32Ty(), SrcArg, Align(1));
  FuncBuilder.CreateAlignedStore(Load0, DstArg, Align(1));
  Value *Src4 = FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), SrcArg,
                                      FuncBuilder.getInt32(4), "src.4");
  Value *Dst4 = FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), DstArg,
                                      FuncBuilder.getInt32(4), "dst.4");
  Value *Load1 =
      FuncBuilder.CreateAlignedLoad(FuncBuilder.getInt32Ty(), Src4, Align(1));
  FuncBuilder.CreateAlignedStore(Load1, Dst4, Align(1));

  Value *RemainingSizeAfter8Bytes = FuncBuilder.CreateSub(
      SizeArg, FuncBuilder.getInt32(8), "remaining.size.after.8bytes");
  Value *DstAfter8Bytes =
      FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), DstArg,
                            FuncBuilder.getInt32(8), "dst.after.8bytes");
  Value *SrcAfter8Bytes =
      FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), SrcArg,
                            FuncBuilder.getInt32(8), "src.after.8bytes");

  FuncBuilder.CreateMemCpy(DstAfter8Bytes, Align(16), SrcAfter8Bytes, Align(1),
                           RemainingSizeAfter8Bytes);

  FuncBuilder.CreateBr(ReturnBB);

  FuncBuilder.SetInsertPoint(ReturnBB);
  FuncBuilder.CreateRetVoid();

  M->eraseFromParent();

  return true;
}

// src unaligned, dst unaligned, size is divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDstUnalignConst(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrcUnalignDst16Var(M, BBI);
}

// src unaligned, dst unaligned, size is variable
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDstUnalignVar(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrcUnalignDst16Var(M, BBI);
}

void RISCVEsp32P4MemIntrinBase::inlineEsp32P4MemCpy() {
  PassBuilder PB;
  // Populate analysis managers and register Polly-specific analyses.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  ModulePassManager MPM;
  MPM.addPass(AlwaysInlinerPass());
  PreservedAnalyses PA = MPM.run(*TheModule, MAM);
}

bool RISCVEsp32P4MemIntrinPass::processMemCpyToSIMD(MemCpyInst *M,
                                                    BasicBlock::iterator &BBI,
                                                    MemCpyType Type) {

  switch (Type) {
  case MemCpyType::Src16_Dst16_Const16:
    return processSrc16Dst16Const16(Type, M, BBI);
  case MemCpyType::Src16_Dst16_Const8:
    return processSrc16Dst16Const8(Type, M, BBI);
  case MemCpyType::Src16_Dst16_OtherConst: {
    if (Len < 16) {
      // Use the specific handler for 16-aligned small constant copies
      // Assuming processSrc16Dst16From1To15Const exists and takes (M, BBI)
      // If it needs Type, add it: processSrc16Dst16From1To15Const(Type, M,
      // BBI);
      return processSrc16Dst16From1To15Const(M, BBI);
    } else if (Len >= 16) {
      return processSrc16Dst16OtherConst(M, BBI);
    }
    return false; // Length is 0 or not constant
  }
  case MemCpyType::Src16_Dst16_Var:
    return processSrc16Dst16Var(M);
  case MemCpyType::Src16_Dst8_Const16:
    return processSrc16Dst8Const16(Type, M, BBI);
  case MemCpyType::Src16_Dst8_Const8:
    return processSrc16Dst8Const8(Type, M, BBI);
  case MemCpyType::Src16_Dst8_OtherConst: {
    if (Len < 16) {

      return processSrc16Dst16From1To15Const(M, BBI);
    } else if (Len >= 16) {
      return processSrc16Dst8OtherConst(M, BBI);
    }
    return false;
  }
  case MemCpyType::Src16_Dst8_Var:
    return processSrc16Dst8Var(M);
  case MemCpyType::Src16_DstUnalign_Const16:
    return processSrc16DstUnalignConst16(M, BBI);
  case MemCpyType::Src16_DstUnalign_Const8:
    return processSrc16DstUnalignConst8(M, BBI);
  case MemCpyType::Src16_DstUnalign_OtherConst:
    if (Len < 16) {
      return processFromSrcUnalignDstUnalign1To15Const(M, BBI);
    } else if (Len >= 16) {
      return processSrc16DstUnalignOtherConst(M, BBI);
    }
    return false;
  case MemCpyType::Src16_DstUnalign_Var:
    return processSrc16DstUnalignVar(M, BBI);
  case MemCpyType::Src8_Dst16_Const16:
    return processSrc8Dst16Const16(Type, M, BBI);
  case MemCpyType::Src8_Dst16_Const8:
    return processSrc8Dst16Const8(Type, M, BBI);
  case MemCpyType::Src8_Dst16_OtherConst:
    if (Len < 16) {
      return processSrc16Dst16From1To15Const(M, BBI);
    } else if (Len >= 16) {
      return processSrc8Dst16OtherConst(M, BBI);
    }
    return false;
  case MemCpyType::Src8_Dst16_Var:
    return processSrc8Dst16Var(M);
  case MemCpyType::Src8_Dst8_Const16:
    return processSrc8Dst8Const16(Type, M, BBI);
  case MemCpyType::Src8_Dst8_Const8:
    return processSrc8Dst8Const8(Type, M, BBI);
  case MemCpyType::Src8_Dst8_OtherConst:
    if (Len < 16) {
      return processSrc16Dst16From1To15Const(M, BBI);
    } else if (Len >= 16) {
      return processSrc8Dst8OtherConst(M, BBI);
    }
    return false;
  case MemCpyType::Src8_Dst8_Var:
    return processSrc8Dst8Var(M);
  case MemCpyType::Src8_DstUnalign_Const16:
    return processSrc8DstUnalignConst16(M, BBI);
  case MemCpyType::Src8_DstUnalign_Const8:
    return processSrc8DstUnalignConst8(M, BBI);
  case MemCpyType::Src8_DstUnalign_OtherConst:
    if (Len < 16) {
      return processFromSrcUnalignDstUnalign1To15Const(M, BBI);
    } else if (Len >= 16) {
      return processSrc8DstUnalignOtherConst(M, BBI);
    }
    return false;
  case MemCpyType::Src8_DstUnalign_Var:
    return processSrc16DstUnalignVar(M, BBI);
  case MemCpyType::SrcUnalign_Dst16_Const16:
    return processSrcUnalignDst16Const16(M, BBI);
  case MemCpyType::SrcUnalign_Dst16_Const8:
    return processSrcUnalignDst16Const8(M, BBI);
  case MemCpyType::SrcUnalign_Dst16_OtherConst:
    if (Len < 16) {
      return processFromSrcUnalignDstUnalign1To15Const(M, BBI);
    } else if (Len >= 16) {
      return processSrcUnalignDst16OtherConst(M, BBI);
    }
    return false;
  case MemCpyType::SrcUnalign_Dst16_Var:
    return processSrcUnalignDst16Var(M, BBI);
  case MemCpyType::SrcUnalign_Dst8_Const16:
    return processSrcUnalignDst8Const16(M, BBI);
  case MemCpyType::SrcUnalign_Dst8_Const8:
    return processSrcUnalignDst8Const8(M, BBI);
  case MemCpyType::SrcUnalign_Dst8_OtherConst:
    if (Len < 16) {
      return processFromSrcUnalignDstUnalign1To15Const(M, BBI);
    } else if (Len >= 16) {
      return processSrcUnalignDst8OtherConst(M, BBI);
    }
    return false;
  case MemCpyType::SrcUnalign_Dst8_Var:
    return processSrcUnalignDst8Var(M, BBI);
  case MemCpyType::SrcUnalign_DstUnalign_Const16:
  case MemCpyType::SrcUnalign_DstUnalign_Const8:
  case MemCpyType::SrcUnalign_DstUnalign_OtherConst:
    if (Len < 16) {
      return processFromSrcUnalignDstUnalign1To15Const(M, BBI);
    } else if (Len >= 16) {
      return processSrcUnalignDstUnalignConst(M, BBI);
    }
    return false;
  case MemCpyType::SrcUnalign_DstUnalign_Var:
    return processSrcUnalignDstUnalignVar(M, BBI);
  }

  return false;
}

/// Executes one iteration of RISCVEsp32P4MemIntrinPass.
bool RISCVEsp32P4MemIntrinPass::iterateOnFunction(Function &F) {
  bool MadeChange = false;
  // Walk all instruction in the function.
  for (BasicBlock &BB : F) {
    for (BasicBlock::iterator BI = BB.begin(), BE = BB.end(); BI != BE;) {
      // Avoid invalidating the iterator.
      Instruction *I = &*BI++;

      bool RepeatInstruction = false;

      if (auto *M = dyn_cast<MemCpyInst>(I)) {
        if (M->isVolatile())
          continue;

        // Add zero size check - if size is 0, directly remove the instruction
        if (ConstantInt *CI = dyn_cast<ConstantInt>(M->getLength())) {
          if (CI->isZero()) {
            M->eraseFromParent();
            MadeChange = true;
            continue;
          }
        }

        // Convert memcpy to vst/vld
        MemCpyType Type = getMemCpyType(M);
        RepeatInstruction = processMemCpyToSIMD(M, BI, Type);
        if (RepeatInstruction)
          MadeChange = true;
      }
    }
    inlineEsp32P4MemCpy();
  }

  return MadeChange;
}

bool RISCVEsp32P4MemIntrinPass::runImpl(Function &F, TargetLibraryInfo *TLI_,
                                        AAResults *AA_, AssumptionCache *AC_,
                                        DominatorTree *DT_,
                                        PostDominatorTree *PDT_,
                                        MemorySSA *MSSA_,
                                        FunctionAnalysisManager &AM) {
  bool MadeChange = false;
  TLI = TLI_;
  AA = AA_;
  AC = AC_;
  DT = DT_;
  PDT = PDT_;
  MSSA = MSSA_;
  MemorySSAUpdater MSSAU_(MSSA_);
  MSSAU = &MSSAU_;

  while (true) {
    if (!iterateOnFunction(F))
      break;
    MadeChange = true;
  }
  if (VerifyMemorySSA)
    MSSA_->verifyMemorySSA();

  return MadeChange;
}

PreservedAnalyses RISCVEsp32P4MemIntrinPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  if (!EnableRISCVEsp32P4MemIntrin)
    return PreservedAnalyses::all();

  TheModule = F.getParent();
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto *AA = &AM.getResult<AAManager>(F);
  auto *AC = &AM.getResult<AssumptionAnalysis>(F);
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *PDT = &AM.getResult<PostDominatorTreeAnalysis>(F);
  auto *MSSA = &AM.getResult<MemorySSAAnalysis>(F);

  bool MadeChange = runImpl(F, &TLI, AA, AC, DT, PDT, &MSSA->getMSSA(), AM);
  if (!MadeChange)
    return PreservedAnalyses::all();
  if (MadeChange) {
    FunctionPassManager FPM;
    // Basic dead code elimination
    FPM.addPass(DCEPass());

    // Simplify control flow graph - merge basic blocks and delete unreachable
    // code
    FPM.addPass(SimplifyCFGPass());

    // Instruction combination - simplify and optimize instruction sequence
    FPM.addPass(InstCombinePass());

    FPM.run(F, AM);
  }

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<MemorySSAAnalysis>();

  return PA;
}