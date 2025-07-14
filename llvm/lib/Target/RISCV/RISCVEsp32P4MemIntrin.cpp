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
#include "llvm/IR/IntrinsicsRISCV.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-esp32-p4-mem-intrin"

// Command line option to enable the RISCVEsp32P4MemIntrin pass
cl::opt<bool> llvm::EnableRISCVEsp32P4MemIntrin(
    "riscv-esp32-p4-mem-intrin", cl::init(false),
    cl::desc("Enable loop unrolling and remainder specific loop"));

static cl::opt<unsigned> MemIntrinUnrollThresholdDefault(
    "riscv-esp32-p4-mem-intrin-unroll-threshold", cl::init(150), cl::Hidden,
    cl::desc("Maximum memcpy size (in bytes) to fully unroll instead of "
             "generating a loop."));

// Common method to check if function exists and create call
bool RISCVEsp32P4MemIntrinBase::useExistingHelperFunction(
    MemCpyInst *M, IRBuilder<> &Builder, const std::string &FuncName,
    Value *DstAddr, Value *SrcAddr, Value *Size) {

  // Check if function exists in module
  if (Function *ExistingFunc = module->getFunction(FuncName)) {
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

  // Check if function exists in module
  if (Function *ExistingFunc = module->getFunction(FuncName)) {
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
  Function *MCFunc =
      Function::Create(FuncTy, GlobalValue::InternalLinkage, FuncName, module);

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

  // Create new function
  Function *MCFunc =
      Function::Create(FuncTy, GlobalValue::InternalLinkage, FuncName, module);

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
Value *RISCVEsp32P4MemIntrin::generateLoadInstructions(IRBuilder<> &Builder,
                                                       Value *SrcAddr,
                                                       MemCpyType Type,
                                                       int index) {
  switch (Type) {
  case MemCpyType::Src16_Dst16_Const16:
  case MemCpyType::Src16_Dst16_Const8:
  case MemCpyType::Src16_Dst8_Const16:
  case MemCpyType::Src16_Dst8_Const8:
    return createEspVld128Ip(Builder, SrcAddr, index);
  case MemCpyType::Src8_Dst16_Const16:
  case MemCpyType::Src8_Dst16_Const8:
  case MemCpyType::Src8_Dst8_Const16:
    SrcAddr = createEspVldL64Ip(Builder, SrcAddr, index);
    return createEspVldH64Ip(Builder, SrcAddr, index);
  case MemCpyType::Src8_Dst8_Const8:
    return createEspVldH64Ip(Builder, SrcAddr, index);
  default:
    return SrcAddr;
  }
}

// Add helper function to handle load/store instruction generation
Value *RISCVEsp32P4MemIntrin::generateStoreInstructions(IRBuilder<> &Builder,
                                                        Value *DstAddr,
                                                        MemCpyType Type,
                                                        int index) {
  switch (Type) {
  case MemCpyType::Src16_Dst16_Const16:
  case MemCpyType::Src16_Dst16_Const8:
  case MemCpyType::Src8_Dst16_Const16:
  case MemCpyType::Src8_Dst16_Const8:
    // Call the new intrinsic function and return its result (updated DstAddr)
    return createEspVst128Ip(Builder, DstAddr, index);
  case MemCpyType::Src16_Dst8_Const16:
  case MemCpyType::Src16_Dst8_Const8:
  case MemCpyType::Src8_Dst8_Const16: {
    // First call vst.l.64.ip to get the updated address
    Value *UpdatedDstAddr = createEspVstL64Ip(Builder, DstAddr, index);
    // Pass the updated address to vst.h.64.ip and return the final updated
    // address
    return createEspVstH64Ip(Builder, UpdatedDstAddr, index);
  }
  case MemCpyType::Src8_Dst8_Const8:
    // Call the new intrinsic function and return its result (updated DstAddr)
    return createEspVstH64Ip(Builder, DstAddr, index);
  default:
    // Return the original address for unhandled cases
    // (or you can consider assert(false, "Unhandled MemCpyType");)
    return DstAddr;
  }
}

// Process a complete data block
void RISCVEsp32P4MemIntrin::processDataBlock(IRBuilder<> &Builder,
                                             Value *&SrcAddr, Value *&DstAddr,
                                             MemCpyType Type, int blockSize) {
  // Use local variables to track the current address in the loop
  Value *CurrentSrc = SrcAddr;
  Value *CurrentDst = DstAddr;

  // Load loop:每次调用都使用上一次返回的地址
  for (int j = 0; j < blockSize; j++) {
    if (j == 0) {
      CurrentSrc = generateLoadInstructions(Builder, SrcAddr, Type, j);
    } else {
      CurrentSrc = generateLoadInstructions(Builder, CurrentSrc, Type, j);
    }
  }

  // Store loop:每次调用都使用上一次返回的地址
  for (int j = 0; j < blockSize; j++) {
    if (j == 0) {
      CurrentDst = generateStoreInstructions(Builder, DstAddr, Type, j);
    } else {
      CurrentDst = generateStoreInstructions(Builder, CurrentDst, Type, j);
    }
  }

  // Update the original pointer variables passed by reference
  SrcAddr = CurrentSrc;
  DstAddr = CurrentDst;
}

Value *RISCVEsp32P4MemIntrin::createEspVld128Ip(IRBuilder<> &Builder,
                                                Value *src, int index) {
  assert(index >= 0 && index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  // Get intrinsic declaration (the return type is a structure, but we care
  // about the returned pointer)
  Function *IntrinsicFunc =
      Intrinsic::getDeclaration(module, Intrinsic::riscv_esp_vld_128_ip, {});
  // Create intrinsic call, it returns the updated src pointer (i32)
  return Builder.CreateCall(
      IntrinsicFunc,
      {src, ConstantInt::get(i32Ty, 16), ConstantInt::get(i32Ty, index)},
      "vld128ip");
}

// Rename and modify: use intrinsic esp.vld.h.64.ip
// Return the updated src pointer (i32)
Value *RISCVEsp32P4MemIntrin::createEspVldH64Ip(IRBuilder<> &Builder,
                                                Value *src, int index) {
  assert(index >= 0 && index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc =
      Intrinsic::getDeclaration(module, Intrinsic::riscv_esp_vld_h_64_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {src, ConstantInt::get(i32Ty, 8), ConstantInt::get(i32Ty, index)},
      "vldh64ip");
}

// Rename and modify: use intrinsic esp.vld.l.64.ip
// Return the updated src pointer (i32)
Value *RISCVEsp32P4MemIntrin::createEspVldL64Ip(IRBuilder<> &Builder,
                                                Value *src, int index) {
  assert(index >= 0 && index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc =
      Intrinsic::getDeclaration(module, Intrinsic::riscv_esp_vld_l_64_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {src, ConstantInt::get(i32Ty, 8), ConstantInt::get(i32Ty, index)},
      "vldl64ip");
}

// Rename and modify: use intrinsic esp.vst.128.ip
// Return the updated dst pointer (i32)
Value *RISCVEsp32P4MemIntrin::createEspVst128Ip(IRBuilder<> &Builder,
                                                Value *dst, int index) {
  assert(index >= 0 && index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc =
      Intrinsic::getDeclaration(module, Intrinsic::riscv_esp_vst_128_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {ConstantInt::get(i32Ty, index), dst, ConstantInt::get(i32Ty, 16)},
      "vst128ip");
}

// Rename and modify: use intrinsic esp.vst.h.64.ip
// Return the updated dst pointer (i32)
Value *RISCVEsp32P4MemIntrin::createEspVstH64Ip(IRBuilder<> &Builder,
                                                Value *dst, int index) {
  assert(index >= 0 && index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc =
      Intrinsic::getDeclaration(module, Intrinsic::riscv_esp_vst_h_64_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {ConstantInt::get(i32Ty, index), dst, ConstantInt::get(i32Ty, 8)},
      "vsth64ip");
}

// Rename and modify: use intrinsic esp.vst.l.64.ip
// Return the updated dst pointer (i32)
Value *RISCVEsp32P4MemIntrin::createEspVstL64Ip(IRBuilder<> &Builder,
                                                Value *dst, int index) {
  assert(index >= 0 && index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc =
      Intrinsic::getDeclaration(module, Intrinsic::riscv_esp_vst_l_64_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {ConstantInt::get(i32Ty, index), dst, ConstantInt::get(i32Ty, 8)},
      "vstl64ip");
}

enum MemCpyType RISCVEsp32P4MemIntrinBase::getMemCpyType(MemCpyInst *M) {
  MaybeAlign SrcAlign = M->getSourceAlign();
  SrcAlignValue = SrcAlign->value();
  // Determine the source alignment category
  SrcAlignment srcAlignCat = SrcAlignment::SrcUnalign;
  if (isDivisibleBy16(SrcAlignValue))
    srcAlignCat = SrcAlignment::Src16;
  else if (isDivisibleBy8(SrcAlignValue))
    srcAlignCat = SrcAlignment::Src8;

  MaybeAlign DstAlign = M->getDestAlign();
  DstAlignValue = DstAlign->value();
  // Determine the destination alignment category
  DstAlignment dstAlignCat = DstAlignment::DstUnalign;
  if (isDivisibleBy16(DstAlignValue))
    dstAlignCat = DstAlignment::Dst16;
  else if (isDivisibleBy8(DstAlignValue))
    dstAlignCat = DstAlignment::Dst8;

  // Determine the length type
  SizeType sizeType = SizeType::Var;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(M->getLength())) {
    Len = CI->getZExtValue();
    if (isDivisibleBy16(Len))
      sizeType = SizeType::Const16;
    else if (isDivisibleBy8(Len))
      sizeType = SizeType::Const8;
    else
      sizeType = SizeType::OtherConst;
  }
  // for size variable, can't inline the function
  else {
    SizeValue = M->getLength();
  }

  // Three-dimensional conditional judgment
  switch (srcAlignCat) {
  case SrcAlignment::Src16:
    switch (dstAlignCat) {
    case DstAlignment::Dst16:
      switch (sizeType) {
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
      switch (sizeType) {
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
      switch (sizeType) {
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
    switch (dstAlignCat) {
    case DstAlignment::Dst16:
      switch (sizeType) {
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
      switch (sizeType) {
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
      switch (sizeType) {
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
    switch (dstAlignCat) {
    case DstAlignment::Dst16:
      switch (sizeType) {
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
      switch (sizeType) {
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
      switch (sizeType) {
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
    const std::string &FuncName, uint64_t blockSize, uint64_t chunkSize) {

  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  uint64_t times = Len / chunkSize;
  Value *SrcAddr = nullptr;
  Value *DstAddr = nullptr;

  // Len exceeds the specified size, need for loop
  if (times > 8) {
    uint64_t totalBlocks = Len / blockSize;
    uint64_t remainder = Len % blockSize;
    times = remainder / chunkSize;
    SrcAddr = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());
    DstAddr = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());

    // When totalBlocks loop count exceeds threshold, do not expand using loop
    if (totalBlocks > MemIntrinUnrollThresholdDefault) {
      // First check if function exists in current module
      if (useExistingHelperFunction(M, Builder, FuncName, DstAddr, SrcAddr,
                                    Builder.getInt32(Len))) {
        return true;
      }

      // Create loop processing function, must not inline, otherwise wrong
      // result
      Function *MCFunc = createMemCpyHelperFunction(
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
          FuncBuilder.CreateICmpULT(Size, FuncBuilder.getInt32(blockSize));
      FuncBuilder.CreateCondBr(Cmp, ForCleanupBB, ForBodyBB);

      FuncBuilder.SetInsertPoint(ForBodyBB);
      PHINode *I = FuncBuilder.CreatePHI(Builder.getInt32Ty(), 2);
      I->addIncoming(FuncBuilder.getInt32(0), EntryBB);

      // Create PHI nodes for source and destination addresses, used to track
      // the current address being processed in the loop
      PHINode *SrcPtrLoop =
          FuncBuilder.CreatePHI(Builder.getInt32Ty(), 2, "src.ptr.loop");
      SrcPtrLoop->addIncoming(Src, EntryBB);
      Value *SrcPtrInit = SrcPtrLoop;
      PHINode *DstPtrLoop =
          FuncBuilder.CreatePHI(Builder.getInt32Ty(), 2, "dst.ptr.loop");
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
      // Create PHI nodes for source and destination addresses, used to track
      // the current address being processed in the cleanup block
      PHINode *SrcPtrCleanup =
          FuncBuilder.CreatePHI(Builder.getInt32Ty(), 2, "src.ptr.cleanup");
      SrcPtrCleanup->addIncoming(Src, EntryBB);
      SrcPtrCleanup->addIncoming(SrcPtrInit, ForBodyBB);
      Value *SrcPtrCleanupInit = SrcPtrCleanup;
      PHINode *DstPtrCleanup =
          FuncBuilder.CreatePHI(Builder.getInt32Ty(), 2, "dst.ptr.cleanup");
      DstPtrCleanup->addIncoming(Dst, EntryBB);
      DstPtrCleanup->addIncoming(DstPtrInit, ForBodyBB);
      Value *DstPtrCleanupInit = DstPtrCleanup;
      // The remaining remainder part is directly generated
      processDataBlock(FuncBuilder, SrcPtrCleanupInit, DstPtrCleanupInit, Type,
                       times);
      FuncBuilder.CreateRetVoid();
      setLoopMetadata(ForBodyBB->getTerminator());

    } else {
      // Fully expand
      for (uint64_t i = 0; i < totalBlocks; i++) {
        processDataBlock(Builder, SrcAddr, DstAddr, Type, 8);
      }

      processDataBlock(Builder, SrcAddr, DstAddr, Type, times);
    }

  } else {
    // Len does not exceed the specified size, can be processed in one go
    SrcAddr = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());
    DstAddr = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());
    // Directly expand to handle small data
    processDataBlock(Builder, SrcAddr, DstAddr, Type, times);
  }

  // Process possible additional remaining parts (e.g. src16dst8const8 and
  // src8dst16const8 at the end of the function) This part needs to be added
  // based on actual conditions
  switch (Type) {
  case MemCpyType::Src16_Dst16_Const8:
  case MemCpyType::Src8_Dst16_Const8:
  case MemCpyType::Src16_Dst8_Const8:
    SrcAddr = createEspVldL64Ip(Builder, SrcAddr, 0);
    DstAddr = createEspVstL64Ip(Builder, DstAddr, 0);
    break;
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
  Type *I32PtrTy = Builder.getInt32Ty(); // Type for asm operands

  uint64_t BytesCopied = 0;

  // If length >= 8, prioritize using 8-byte copy
  if (Len >= 8) {
    Value *SrcInt = Builder.CreatePtrToInt(CurrentSrc, I32PtrTy);
    Value *DstInt = Builder.CreatePtrToInt(CurrentDst, I32PtrTy);

    // Placeholder for the actual helper function calls
    // Replace these with the actual function names if they differ slightly
    SrcInt =
        createEspVldL64Ip(Builder, SrcInt,
                          0); // Generates esp.vld.l.64.ip q0, $0, 8 with +{a1}
    DstInt =
        createEspVstL64Ip(Builder, DstInt,
                          0); // Generates esp.vst.l.64.ip q0, $0, 8 with +{a0}

    BytesCopied = 8;
    // Update Src/Dst pointers to point to the beginning of the remaining part
    CurrentSrc =
        Builder.CreateGEP(I8Ty, OrigSrc, Builder.getInt32(BytesCopied));
    CurrentDst =
        Builder.CreateGEP(I8Ty, OrigDst, Builder.getInt32(BytesCopied));
  }

  // --- Use LLVM IR to handle remaining bytes (Len - BytesCopied) ---

  // Handle remaining 4 bytes
  if (Len - BytesCopied >= 4) {
    // Note: No need to calculate Offset here, as CurrentSrc/CurrentDst
    // already point to the correct position
    Value *LoadVal = Builder.CreateAlignedLoad(I32Ty, CurrentSrc, Align(4));
    Builder.CreateAlignedStore(LoadVal, CurrentDst, Align(4));
    // Update pointers and counter
    CurrentSrc = Builder.CreateGEP(I8Ty, CurrentSrc, Builder.getInt32(4));
    CurrentDst = Builder.CreateGEP(I8Ty, CurrentDst, Builder.getInt32(4));
    BytesCopied += 4;
  }

  // Handle remaining 2 bytes
  if (Len - BytesCopied >= 2) {
    Value *LoadVal = Builder.CreateAlignedLoad(I16Ty, CurrentSrc, Align(2));
    Builder.CreateAlignedStore(LoadVal, CurrentDst, Align(2));
    // Update pointers and counter
    CurrentSrc = Builder.CreateGEP(I8Ty, CurrentSrc, Builder.getInt32(2));
    CurrentDst = Builder.CreateGEP(I8Ty, CurrentDst, Builder.getInt32(2));
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

// it supports 1-15 bytes
// src  unalign and dst  unalign
bool RISCVEsp32P4MemIntrinPass::processFromSrcUnalignDstUnalign1To15Const(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return false;
}

// Split len into multiples of 16 and remainder
bool RISCVEsp32P4MemIntrinPass::processOtherConstAlign(MemCpyInst *M,
                                                       BasicBlock::iterator &BI,
                                                       uint64_t dstAlign,
                                                       uint64_t srcAlign) {
  uint64_t remainder = Len % 16;
  uint64_t mainSize = Len - remainder;

  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();
  Builder.CreateMemCpy(Dst, Align(dstAlign), Src, Align(srcAlign), mainSize);

  Value *NewSrc =
      Builder.CreateGEP(Builder.getInt8Ty(), Src, Builder.getInt64(mainSize));
  Value *NewDst =
      Builder.CreateGEP(Builder.getInt8Ty(), Dst, Builder.getInt64(mainSize));

  Builder.CreateMemCpy(NewDst, Align(dstAlign), NewSrc, Align(srcAlign),
                       remainder);

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
  return false;
}

// src unalign, dst 8-byte aligned, size is other constant
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst8OtherConst(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return false;
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

  LLVMContext &ctx = Builder.getContext();
  // Check if the function already exists in the current module
  if (useExistingHelperFunction(Builder, FuncName, Dst, Src, Size)) {
    return;
  }

  // Create the helper function signature (ptr, ptr, i32) -> void
  Function *MemCpyFunc = createMemCpyHelperFunctionPtr(Builder, FuncName, Dst,
                                                       Src, Size, isInline);

  // Create basic blocks for the helper function
  BasicBlock *EntryBB = BasicBlock::Create(ctx, "entry", MemCpyFunc);
  BasicBlock *ReturnBB = BasicBlock::Create(ctx, "return",
                                            MemCpyFunc); // Common exit block

  // Create basic blocks for each case in the switch
  std::vector<BasicBlock *> SwitchBBs;
  for (int i = 1; i <= 15; i++) {
    SwitchBBs.push_back(
        BasicBlock::Create(ctx, "sw.bb" + std::to_string(i), MemCpyFunc));
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
  for (int i = 1; i <= 15; i++) {
    // Add case: if SizeArg == i, jump to SwitchBBs[i-1]
    SI->addCase(FuncBuilder.getInt32(i), SwitchBBs[i - 1]);
  }

  // --- Populate Switch Case Basic Blocks with LLVM IR ---
  llvm::Type *I8Ty = FuncBuilder.getInt8Ty();
  llvm::Type *I16Ty = FuncBuilder.getInt16Ty();
  llvm::Type *I32Ty = FuncBuilder.getInt32Ty();

  for (int i = 1; i <= 15; i++) {
    FuncBuilder.SetInsertPoint(
        SwitchBBs[i - 1]); // Set builder to the correct case block
    uint64_t BytesToCopy = i;
    uint64_t BytesCopied = 0;
    if (Type == MemCpyType::Src16_Dst16_Var ||
        Type == MemCpyType::Src16_Dst8_Var ||
        Type == MemCpyType::Src8_Dst16_Var ||
        Type == MemCpyType::Src8_Dst8_Var) {
      if (BytesToCopy - BytesCopied >= 8) {
        Value *SrcInt = FuncBuilder.CreatePtrToInt(SrcArg, I32Ty);
        Value *DstInt = FuncBuilder.CreatePtrToInt(DstArg, I32Ty);

        // Placeholder for the actual helper function calls
        // Replace these with the actual function names if they differ slightly
        SrcInt = createEspVldL64Ip(
            FuncBuilder, SrcInt,
            0); // Generates esp.vld.l.64.ip q0, $0, 8 with +{a1}
        DstInt = createEspVstL64Ip(
            FuncBuilder, DstInt,
            0); // Generates esp.vst.l.64.ip q0, $0, 8 with +{a0}

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

  LLVMContext &ctx = Builder.getContext();
  // Check if the function already exists in the current module
  if (useExistingHelperFunction(Builder, FuncName, Dst, Src, Size)) {
    return;
  }

  // Create the helper function signature (ptr, ptr, i32) -> void
  Function *MemCpyFunc = createMemCpyHelperFunctionPtr(Builder, FuncName, Dst,
                                                       Src, Size, isInline);

  // Create basic blocks for the helper function
  BasicBlock *EntryBB = BasicBlock::Create(ctx, "entry", MemCpyFunc);
  BasicBlock *ReturnBB = BasicBlock::Create(ctx, "return",
                                            MemCpyFunc); // Common exit block

  // Create basic blocks for each case in the switch
  std::vector<BasicBlock *> SwitchBBs;
  for (int i = 1; i <= 7; i++) {
    SwitchBBs.push_back(
        BasicBlock::Create(ctx, "sw.bb" + std::to_string(i), MemCpyFunc));
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
  for (int i = 1; i <= 7; i++) {
    // Add case: if SizeArg == i, jump to SwitchBBs[i-1]
    SI->addCase(FuncBuilder.getInt32(i), SwitchBBs[i - 1]);
  }

  // --- Populate Switch Case Basic Blocks with LLVM IR ---
  llvm::Type *I8Ty = FuncBuilder.getInt8Ty();
  llvm::Type *I16Ty = FuncBuilder.getInt16Ty();
  llvm::Type *I32Ty = FuncBuilder.getInt32Ty();

  for (int i = 1; i <= 7; i++) {
    FuncBuilder.SetInsertPoint(
        SwitchBBs[i - 1]); // Set builder to the correct case block
    uint64_t BytesToCopy = i;
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
  return false;
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
  // Generate unique function name
  std::string FuncName = "esp32p4MemCpySrc16DstunalignVar";

  if (Function *ExistingFunc = module->getFunction(FuncName)) {
    // If function exists, create call directly
    Builder.CreateCall(ExistingFunc,
                       {Dst, Src, Size, Builder.getInt32(DstAlignValue)});
    M->eraseFromParent();
    return true;
  }

  // Create new function type
  FunctionType *FuncTy =
      FunctionType::get(Builder.getVoidTy(),
                        {Builder.getPtrTy(), Builder.getPtrTy(),
                         Builder.getInt32Ty(), Builder.getInt32Ty()},
                        false);

  Function *MemCpyFunc = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                          FuncName, M->getModule());

  Value *Args[] = {Dst, Src, Size, Builder.getInt32(DstAlignValue)}; // TODO

  // Create a tail call to MemCpyFunction
  CallInst *TailCall =
      CallInst::Create(MemCpyFunc->getFunctionType(), MemCpyFunc, Args, "",
                       /*InsertBefore=*/nullptr);
  TailCall->setTailCallKind(CallInst::TCK_Tail);
  Builder.Insert(TailCall);
  // Get Function* from FunctionCallee
  assert(MemCpyFunc && "esp32p4MemCpySrc16DstunalignVar must be a function!");

  // Set function attributes
  MemCpyFunc->addFnAttr(Attribute::NoUnwind);
  MemCpyFunc->addFnAttr(Attribute::NoInline);

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
  Value *DstValueArg = MemCpyFunc->arg_begin() + 3;
  DstValueArg->setName("dst_align");
  // Create basic blocks
  BasicBlock *IfThenBB =
      BasicBlock::Create(M->getContext(), "if.then", MemCpyFunc);
  BasicBlock *IfEndBB =
      BasicBlock::Create(M->getContext(), "if.end", MemCpyFunc);
  BasicBlock *CleanupBB =
      BasicBlock::Create(M->getContext(), "cleanup", MemCpyFunc);

  // Entry block
  Value *Sub =
      FuncBuilder.CreateSub(FuncBuilder.getInt32(16), DstValueArg, "sub");
  Value *Cmp = FuncBuilder.CreateICmpULT(Sub, SizeArg, "cmp.not");
  FuncBuilder.CreateCondBr(Cmp, IfEndBB, IfThenBB);

  // if.then block
  FuncBuilder.SetInsertPoint(IfThenBB);
  processMemCpyVarFrom1To15(
      FuncBuilder, "esp32p4MemCpySrcUnalignDstUnalignFrom1To15Opt", DstArg,
      SrcArg, SizeArg, true, MemCpyType::SrcUnalign_DstUnalign_Var);
  FuncBuilder.CreateBr(CleanupBB);

  // if.end block
  FuncBuilder.SetInsertPoint(IfEndBB);
  processMemCpyVarFrom1To15(
      FuncBuilder, "esp32p4MemCpySrcUnalignDstUnalignFrom1To15Opt", DstArg,
      SrcArg, Sub, true, MemCpyType::SrcUnalign_DstUnalign_Var);
  Value *Sub2 = FuncBuilder.CreateSub(SizeArg, Sub, "sub2");
  Value *AddPtr1 =
      FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), DstArg, Sub, "add.ptr1");
  Value *AddPtr =
      FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), SrcArg, Sub, "add.ptr");

  FuncBuilder.CreateMemCpy(AddPtr1, Align(16), AddPtr, Align(1), Sub2);
  FuncBuilder.CreateBr(CleanupBB);

  // Return block
  FuncBuilder.SetInsertPoint(CleanupBB);
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
Value *RISCVEsp32P4MemIntrin::createEspLd128UsarIp(IRBuilder<> &Builder,
                                                   Value *src, int index) {
  assert(index >= 0 && index <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc = Intrinsic::getDeclaration(
      module, Intrinsic::riscv_esp_ld_128_usar_ip, {});
  return Builder.CreateCall(
      IntrinsicFunc,
      {src, ConstantInt::get(i32Ty, 16), ConstantInt::get(i32Ty, index)},
      "ld128usarip");
}

// Return the updated src pointer (i32)
Value *RISCVEsp32P4MemIntrin::createEspSrcQLdIp(IRBuilder<> &Builder,
                                                Value *src, int index0,
                                                int index2, int index3,
                                                int index4) {
  assert(index0 >= 0 && index0 <= 7 && "Index must be between 0 and 7");
  assert(index3 >= 0 && index3 <= 7 && "Index must be between 0 and 7");
  assert(index4 >= 0 && index4 <= 7 && "Index must be between 0 and 7");
  Type *i32Ty = Builder.getInt32Ty();
  Function *IntrinsicFunc =
      Intrinsic::getDeclaration(module, Intrinsic::riscv_esp_src_q_ld_ip, {});
  // Intrinsic arguments: ptr, imm, q_idx1, q_idx2, q_idx_dst
  return Builder.CreateCall(
      IntrinsicFunc,
      {ConstantInt::get(i32Ty, index4), src, ConstantInt::get(i32Ty, index3),
       ConstantInt::get(i32Ty, index2), ConstantInt::get(i32Ty, index0)},
      "srcqldip");
}

// No pointer returned, pure calculation instructions
void inline RISCVEsp32P4MemIntrin::createEspSrcQ(IRBuilder<> &Builder,
                                                 int index0, int index1,
                                                 int index2) {
  assert(index0 >= 0 && index0 <= 7 && "Index must be between 0 and 7");
  assert(index1 >= 0 && index1 <= 7 && "Index must be between 0 and 7");
  assert(index2 >= 0 && index2 <= 7 && "Index must be between 0 and 7");
  Function *IntrinsicFunc =
      Intrinsic::getDeclaration(module, Intrinsic::riscv_esp_src_q);
  Type *i32Ty = Builder.getInt32Ty();
  // Intrinsic arguments: q_idx1, q_idx2, q_idx_dst
  Builder.CreateCall(IntrinsicFunc, {ConstantInt::get(i32Ty, index2),
                                     ConstantInt::get(i32Ty, index1),
                                     ConstantInt::get(i32Ty, index0)});
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstDiv48(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t quotient) {
  return false;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstMod48From32To47(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t quotient,
    uint64_t remainder) {
  return false;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstMod48From16To31(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t quotient,
    uint64_t remainder) {
  return false;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstMod48From1To15(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t quotient,
    uint64_t remainder) {
  return false;
}
// src unaligned, dst 16-byte aligned, size is divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16Const16(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return false;
}

// src unaligned, dst 16-byte aligned, size is divisible by 8
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16Const8(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return false;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16Common(
    MemCpyInst *M, BasicBlock::iterator &BBI, bool isInline) {
  return false;
}

// src unaligned, dst 16-byte aligned, size is variable
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16Var(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return processSrcUnalignDst16Common(M, BBI, true);
}

// src unaligned, dst 8-byte aligned, size is divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst8Const16(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return false;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst8Const8(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return false;
}

// src unaligned, dst 8-byte aligned, size is variable
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst8Var(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  return false;
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
  PreservedAnalyses PA = MPM.run(*module, MAM);
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
    if (Len > 0 && Len < 16) {
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
    if (Len > 0 && Len < 16) {

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
    if (Len > 0 && Len < 16) {
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
    if (Len > 0 && Len < 16) {
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
    if (Len > 0 && Len < 16) {
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
    if (Len > 0 && Len < 16) {
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
    if (Len > 0 && Len < 16) {
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
    if (Len > 0 && Len < 16) {
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
    if (Len > 0 && Len < 16) {
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

bool RISCVEsp32P4MemIntrinPass::runImpl(
    Function &F, TargetLibraryInfo *TLI_, AliasAnalysis *AA_,
    AssumptionCache *AC_, DominatorTree *DT_, PostDominatorTree *PDT_,
    MemorySSA *MSSA_, FunctionAnalysisManager &AM) {
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

  module = F.getParent();
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