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
  for (int J = 0; J < blockSize; J++) {
    if (J == 0) {
      CurrentSrc = generateLoadInstructions(Builder, SrcAddr, Type, J);
    } else {
      CurrentSrc = generateLoadInstructions(Builder, CurrentSrc, Type, J);
    }
  }

  // Store loop:每次调用都使用上一次返回的地址
  for (int J = 0; J < blockSize; J++) {
    if (J == 0) {
      CurrentDst = generateStoreInstructions(Builder, DstAddr, Type, J);
    } else {
      CurrentDst = generateStoreInstructions(Builder, CurrentDst, Type, J);
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
      for (uint64_t I = 0; I < totalBlocks; I++) {
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
  for (int I = 1; I <= 15; I++) {
    SwitchBBs.push_back(
        BasicBlock::Create(ctx, "sw.bb" + std::to_string(I), MemCpyFunc));
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
  for (int I = 1; I <= 7; I++) {
    SwitchBBs.push_back(
        BasicBlock::Create(ctx, "sw.bb" + std::to_string(I), MemCpyFunc));
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
  Value *SrcAddr = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());
  Value *DstAddr = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());

  std::string FuncName = "esp32p4MemCpy" + srcdstcase + "Var";

  if (useExistingHelperFunction(M, Builder, FuncName, DstAddr, SrcAddr, Size)) {
    return true;
  }

  Function *MemCpyFunc = createMemCpyHelperFunction(Builder, FuncName, DstAddr,
                                                    SrcAddr, Size, false);

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
  Value *DstPtr = FuncBuilder.CreateIntToPtr(DstArg, Builder.getPtrTy());
  Value *SrcPtr = FuncBuilder.CreateIntToPtr(SrcArg, Builder.getPtrTy());
  std::string FuncName1_7 = "esp32p4MemCpy" + srcdstcase + "From1To7Opt";
  processMemCpyVarFrom1To7(FuncBuilder, FuncName1_7, DstPtr, SrcPtr, SizeArg,
                           false);
  FuncBuilder.CreateBr(ReturnBB);

  FuncBuilder.SetInsertPoint(CheckMidSizeRange);
  Value *IsLT16 =
      FuncBuilder.CreateICmpULT(SizeArg, FuncBuilder.getInt32(16), "is.lt.16");
  FuncBuilder.CreateCondBr(IsLT16, HandleMidSize, HandleLargeSizeLoop);

  FuncBuilder.SetInsertPoint(HandleMidSize);
  SrcArg = createEspVldL64Ip(FuncBuilder, SrcArg, 0);
  DstArg = createEspVstL64Ip(FuncBuilder, DstArg, 0);
  LLVMContext &ctx = M->getContext();
  PointerType *ptrTy = PointerType::getUnqual(ctx);
  Value *DstPtr2 = FuncBuilder.CreateIntToPtr(DstArg, ptrTy);
  Value *SrcPtr2 = FuncBuilder.CreateIntToPtr(SrcArg, ptrTy);
  Value *SizeMinus8 = FuncBuilder.CreateAdd(SizeArg, FuncBuilder.getInt32(-8),
                                            "size.minus.8", false, true);
  processMemCpyVarFrom1To7(FuncBuilder, FuncName1_7, DstPtr2, SrcPtr2,
                           SizeMinus8, true);
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
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "src.ptr.loop");
  SrcPtrInLoop->addIncoming(SrcArgOrg, HandleLargeSizeLoop);
  PHINode *DstPtrInLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "dst.ptr.loop");
  DstPtrInLoop->addIncoming(DstArgOrg, HandleLargeSizeLoop);

  for (int I = 0; I < 8; I++) {
    if (SrcAlign == 16) {
      SrcArg =
          createEspVld128Ip(FuncBuilder, I == 0 ? SrcPtrInLoop : SrcArg, I);
    } else {
      SrcArg =
          createEspVldL64Ip(FuncBuilder, I == 0 ? SrcPtrInLoop : SrcArg, I);
      SrcArg = createEspVldH64Ip(FuncBuilder, SrcArg, I);
    }
  }
  SrcPtrInLoop->addIncoming(SrcArg, LoopBody128B);

  for (int I = 0; I < 8; I++) {
    if (DstAlign == 16) {
      DstArg =
          createEspVst128Ip(FuncBuilder, I == 0 ? DstPtrInLoop : DstArg, I);
    } else {
      DstArg =
          createEspVstL64Ip(FuncBuilder, I == 0 ? DstPtrInLoop : DstArg, I);
      DstArg = createEspVstH64Ip(FuncBuilder, DstArg, I);
    }
  }
  DstPtrInLoop->addIncoming(DstArg, LoopBody128B);

  Value *LoopNext = FuncBuilder.CreateAdd(LoopIndex, FuncBuilder.getInt32(1),
                                          "loop.inc", true, true);
  LoopIndex->addIncoming(LoopNext, LoopBody128B);
  Value *IsLoopDone =
      FuncBuilder.CreateICmpEQ(LoopNext, Num128BBlocks, "loop.done");
  FuncBuilder.CreateCondBr(IsLoopDone, LoopExitCleanup, LoopBody128B);

  FuncBuilder.SetInsertPoint(LoopExitCleanup);
  PHINode *SrcPtrAfterLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "src.ptr.after.loop");
  SrcPtrAfterLoop->addIncoming(SrcArgOrg, HandleLargeSizeLoop);
  SrcPtrAfterLoop->addIncoming(SrcArg, LoopBody128B);
  PHINode *DstPtrAfterLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "dst.ptr.after.loop");
  DstPtrAfterLoop->addIncoming(DstArgOrg, HandleLargeSizeLoop);
  DstPtrAfterLoop->addIncoming(DstArg, LoopBody128B);

  SwitchInst *Switch = FuncBuilder.CreateSwitch(Remaining16B, InvalidCaseTrap);
  FuncBuilder.SetInsertPoint(HandleTailBlockSwitch);

  PHINode *SrcPtrInTailSwitch =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "src.ptr.tail");
  SrcPtrInTailSwitch->addIncoming(SrcPtrAfterLoop, LoopExitCleanup);
  PHINode *DstPtrInTailSwitch =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "dst.ptr.tail");
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

    for (int J = 0; J < I; J++) {
      SrcArg = (SrcAlign == 16)
                   ? createEspVld128Ip(FuncBuilder,
                                       J == 0 ? SrcPtrAfterLoop : SrcArg, J)
                   : createEspVldH64Ip(
                         FuncBuilder,
                         createEspVldL64Ip(
                             FuncBuilder, J == 0 ? SrcPtrAfterLoop : SrcArg, J),
                         J);
    }
    SrcPtrInTailSwitch->addIncoming(SrcArg, CaseBB);

    for (int J = 0; J < I; J++) {
      DstArg = (DstAlign == 16)
                   ? createEspVst128Ip(FuncBuilder,
                                       J == 0 ? DstPtrAfterLoop : DstArg, J)
                   : createEspVstH64Ip(
                         FuncBuilder,
                         createEspVstL64Ip(
                             FuncBuilder, J == 0 ? DstPtrAfterLoop : DstArg, J),
                         J);
    }
    DstPtrInTailSwitch->addIncoming(DstArg, CaseBB);
    FuncBuilder.CreateBr(HandleTailBlockSwitch);
  }
  Switch->addCase(FuncBuilder.getInt32(0), HandleTailBlockSwitch);

  FuncBuilder.SetInsertPoint(Handle8ByteTail);
  SrcArg = createEspVldL64Ip(FuncBuilder, SrcPtrInTailSwitch, 0);
  DstArg = createEspVstL64Ip(FuncBuilder, DstPtrInTailSwitch, 0);
  FuncBuilder.CreateBr(After8ByteTail);

  FuncBuilder.SetInsertPoint(After8ByteTail);
  PHINode *SrcPtrAfter8B =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "src.ptr.after.8B");
  SrcPtrAfter8B->addIncoming(SrcPtrInTailSwitch, HandleTailBlockSwitch);
  SrcPtrAfter8B->addIncoming(SrcArg, Handle8ByteTail);
  PHINode *DstPtrAfter8B =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "dst.ptr.after.8B");
  DstPtrAfter8B->addIncoming(DstPtrInTailSwitch, HandleTailBlockSwitch);
  DstPtrAfter8B->addIncoming(DstArg, Handle8ByteTail);
  Value *HasRemainingBytes =
      FuncBuilder.CreateICmpEQ(RemainingBytes, FuncBuilder.getInt32(0));
  BasicBlock *HandleRemainingBytes =
      BasicBlock::Create(M->getContext(), "handle.remaining.bytes", MemCpyFunc);
  FuncBuilder.CreateCondBr(HasRemainingBytes, ReturnBB, HandleRemainingBytes);

  FuncBuilder.SetInsertPoint(HandleRemainingBytes);
  Value *SrcFinal = FuncBuilder.CreateIntToPtr(SrcPtrAfter8B, ptrTy);
  Value *DstFinal = FuncBuilder.CreateIntToPtr(DstPtrAfter8B, ptrTy);
  processMemCpyVarFrom1To7(FuncBuilder, FuncName1_7, DstFinal, SrcFinal,
                           RemainingBytes, true);
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

  if (Function *ExistingFunc = module->getFunction(FuncName)) {
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
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  Value *SrcAddr = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());
  Value *DstAddr = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());
  // In each function, the loop count is different, so it must be different
  static int FuncCounter = 0;
  std::string FuncName = "esp32p4MemCpySrcunalignedDst16Div48Index" +
                         std::to_string(FuncCounter++);

  // Create new function type
  FunctionType *FuncTy = FunctionType::get(
      Builder.getVoidTy(), {Builder.getInt32Ty(), Builder.getInt32Ty()}, false);

  // Create new function
  Function *MemCpyFunc = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                          FuncName, M->getModule());

  Value *Args[] = {DstAddr, SrcAddr};

  // Create a tail call to MemCpyFunction
  CallInst *TailCall =
      CallInst::Create(MemCpyFunc->getFunctionType(), MemCpyFunc, Args, "",
                       /*InsertBefore=*/nullptr);
  TailCall->setTailCallKind(CallInst::TCK_Tail);
  Builder.Insert(TailCall);
  // 通过 dyn_cast 从 FunctionCallee 中取出 Function*
  assert(MemCpyFunc && "must be a function!");

  // Extract function arguments
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");

  BasicBlock *EntryBB = nullptr, *ForBodyBB = nullptr,
             *ForCondCleanupBB = nullptr;
  createLoopBlocks(MemCpyFunc, EntryBB, ForBodyBB, ForCondCleanupBB);

  IRBuilder<> FuncBuilder(EntryBB);

  createEspLd128UsarIpAsm(FuncBuilder, SrcArg, 0);
  createEspLd128UsarIpAsm(FuncBuilder, SrcArg, 1);
  FuncBuilder.CreateBr(ForBodyBB);

  FuncBuilder.SetInsertPoint(ForCondCleanupBB);
  FuncBuilder.CreateRetVoid();

  FuncBuilder.SetInsertPoint(ForBodyBB);
  PHINode *I = FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "i.08");
  I->addIncoming(FuncBuilder.getInt32(0), EntryBB);

  Value *Inc =
      FuncBuilder.CreateAdd(I, FuncBuilder.getInt32(1), "inc", true, true);
  I->addIncoming(Inc, ForBodyBB);

  createEspSrcQLdIpAsm(FuncBuilder, SrcArg, 2, 16, 0, 1);
  createEspVst128IpAsm(FuncBuilder, DstArg, 0);

  // Second group of operations
  createEspSrcQLdIpAsm(FuncBuilder, SrcArg, 0, 16, 1, 2);
  createEspVst128IpAsm(FuncBuilder, DstArg, 1);

  // Third group of operations
  createEspSrcQLdIpAsm(FuncBuilder, SrcArg, 1, 16, 2, 0);
  createEspVst128IpAsm(FuncBuilder, DstArg, 2);

  Value *ExitCond = FuncBuilder.CreateICmpEQ(
      Inc, FuncBuilder.getInt32(quotient), "exitcond.not");
  FuncBuilder.CreateCondBr(ExitCond, ForCondCleanupBB, ForBodyBB);

  M->eraseFromParent();

  return true;
}


bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstMod48From32To47(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t quotient,
    uint64_t remainder) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  Value *SrcAddr = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());
  Value *DstAddr = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());
  static int FuncCounter = 0;
  std::string FuncName = "esp32p4MemCpySrcunalignedDst16mod48From32To47Index" +
                         std::to_string(FuncCounter++);

  Function *MemCpyFunc =
      createMemCpyHelperFunction(Builder, FuncName, DstAddr, SrcAddr, false);
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");

  BasicBlock *EntryBB = nullptr, *ForBodyBB = nullptr,
             *ForCondCleanupBB = nullptr;
  createLoopBlocks(MemCpyFunc, EntryBB, ForBodyBB, ForCondCleanupBB);

  IRBuilder<> FuncBuilder(EntryBB);

  SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 0);
  SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 1);
  FuncBuilder.CreateBr(ForBodyBB);

  FuncBuilder.SetInsertPoint(ForBodyBB);
  PHINode *I = FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "i.013");
  I->addIncoming(FuncBuilder.getInt32(0), EntryBB);

  // Create PHI node for source pointer
  PHINode *SrcPtrLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "src.ptr.loop");
  SrcPtrLoop->addIncoming(SrcArg, EntryBB);

  // Create PHI node for destination pointer
  PHINode *DstPtrLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "dst.ptr.loop");
  DstPtrLoop->addIncoming(DstArg, EntryBB);

  Value *Inc =
      FuncBuilder.CreateAdd(I, FuncBuilder.getInt32(1), "inc", true, true);
  I->addIncoming(Inc, ForBodyBB);

  // First group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcPtrLoop, 2, 16, 0, 1);
  DstArg = createEspVst128Ip(FuncBuilder, DstPtrLoop, 0);

  // Second group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcArg, 0, 16, 1, 2);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 1);

  // Third group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcArg, 1, 16, 2, 0);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 2);

  // Update PHI nodes at the end of the loop
  SrcPtrLoop->addIncoming(SrcArg, ForBodyBB);
  DstPtrLoop->addIncoming(DstArg, ForBodyBB);

  Value *ExitCond = FuncBuilder.CreateICmpEQ(
      Inc, FuncBuilder.getInt32(quotient), "exitcond.not");
  FuncBuilder.CreateCondBr(ExitCond, ForCondCleanupBB, ForBodyBB);

  FuncBuilder.SetInsertPoint(ForCondCleanupBB);

  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcArg, 2, 0, 0, 1);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 0);
  createEspSrcQ(FuncBuilder, 1, 1, 2);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 1);
  Value *adjusted_src_ptr = FuncBuilder.CreateAdd(
      SrcArg, FuncBuilder.getInt32(-32), "adjusted_src_ptr");

  Value *DstPtr = FuncBuilder.CreateIntToPtr(DstArg, FuncBuilder.getPtrTy());
  Value *SrcPtr =
      FuncBuilder.CreateIntToPtr(adjusted_src_ptr, FuncBuilder.getPtrTy());

  FuncBuilder.CreateMemCpy(DstPtr, Align(1), SrcPtr, Align(1),
                           FuncBuilder.getInt32(remainder - 32));

  FuncBuilder.CreateRetVoid();

  M->eraseFromParent();
  return true;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstMod48From16To31(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t quotient,
    uint64_t remainder) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  Value *SrcAddr = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());
  Value *DstAddr = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());
  static int FuncCounter = 0;
  std::string FuncName = "esp32p4MemCpySrcunalignedDst16mod48From16to31." +
                         std::to_string(FuncCounter++);

  FunctionType *FuncTy = FunctionType::get(
      Builder.getVoidTy(), {Builder.getInt32Ty(), Builder.getInt32Ty()}, false);

  Function *MemCpyFunc = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                          FuncName, M->getModule());

  Value *Args[] = {DstAddr, SrcAddr};

  CallInst *TailCall =
      CallInst::Create(MemCpyFunc->getFunctionType(), MemCpyFunc, Args, "",
                       /*InsertBefore=*/nullptr);
  TailCall->setTailCallKind(CallInst::TCK_Tail);
  Builder.Insert(TailCall);

  assert(MemCpyFunc && "esp32p4_memcpy_128 must be a function!");

  MemCpyFunc->addFnAttr(Attribute::NoUnwind);

  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");

  BasicBlock *EntryBB = nullptr, *ForBodyBB = nullptr,
             *ForCondCleanupBB = nullptr;
  createLoopBlocks(MemCpyFunc, EntryBB, ForBodyBB, ForCondCleanupBB);

  IRBuilder<> FuncBuilder(EntryBB);

  SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 0);
  SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 1);
  FuncBuilder.CreateBr(ForBodyBB);

  FuncBuilder.SetInsertPoint(ForBodyBB);
  PHINode *I = FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "i.013");
  I->addIncoming(FuncBuilder.getInt32(0), EntryBB);

  // Create PHI node for source pointer
  PHINode *SrcPtrLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "src.ptr.loop");
  SrcPtrLoop->addIncoming(SrcArg, EntryBB);

  // Create PHI node for destination pointer
  PHINode *DstPtrLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "dst.ptr.loop");
  DstPtrLoop->addIncoming(DstArg, EntryBB);

  Value *Inc =
      FuncBuilder.CreateAdd(I, FuncBuilder.getInt32(1), "inc", true, true);
  I->addIncoming(Inc, ForBodyBB);

  // First group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcPtrLoop, 2, 16, 0, 1);
  DstArg = createEspVst128Ip(FuncBuilder, DstPtrLoop, 0);

  // Second group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcArg, 0, 16, 1, 2);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 1);

  // Third group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcArg, 1, 16, 2, 0);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 2);

  // Update PHI nodes at the end of the loop
  SrcPtrLoop->addIncoming(SrcArg, ForBodyBB);
  DstPtrLoop->addIncoming(DstArg, ForBodyBB);

  Value *ExitCond = FuncBuilder.CreateICmpEQ(
      Inc, FuncBuilder.getInt32(quotient), "exitcond.not");
  FuncBuilder.CreateCondBr(ExitCond, ForCondCleanupBB, ForBodyBB);

  FuncBuilder.SetInsertPoint(ForCondCleanupBB);

  createEspSrcQ(FuncBuilder, 0, 0, 1);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 0);
  // Adjust the source pointer, subtract 32 bytes
  Value *adjusted_src_ptr = FuncBuilder.CreateAdd(
      SrcArg, FuncBuilder.getInt32(-32), "adjusted_src_ptr");
  Value *DstPtr = FuncBuilder.CreateIntToPtr(DstArg, FuncBuilder.getPtrTy());
  Value *SrcPtr =
      FuncBuilder.CreateIntToPtr(adjusted_src_ptr, FuncBuilder.getPtrTy());

  FuncBuilder.CreateMemCpy(DstPtr, Align(1), SrcPtr, Align(1),
                           FuncBuilder.getInt32(remainder - 16));

  FuncBuilder.CreateRetVoid();

  M->eraseFromParent();
  return true;
}

bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16ConstMod48From1To15(
    MemCpyInst *M, BasicBlock::iterator &BBI, uint64_t quotient,
    uint64_t remainder) {
  IRBuilder<> Builder(M);
  Value *Src = M->getSource();
  Value *Dst = M->getDest();

  Value *SrcAddr = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());
  Value *DstAddr = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());
  static int FuncCounter = 0;
  std::string FuncName = "esp32p4_memcpy_srcunaligned_dst16mod48_1to15Index" +
                         std::to_string(FuncCounter++);

  Function *MemCpyFunc =
      createMemCpyHelperFunction(Builder, FuncName, DstAddr, SrcAddr);
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");

  BasicBlock *EntryBB = nullptr, *ForBodyBB = nullptr,
             *ForCondCleanupBB = nullptr;
  createLoopBlocks(MemCpyFunc, EntryBB, ForBodyBB, ForCondCleanupBB);

  IRBuilder<> FuncBuilder(EntryBB);

  SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 0);
  SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 1);
  FuncBuilder.CreateBr(ForBodyBB);

  FuncBuilder.SetInsertPoint(ForBodyBB);
  PHINode *I = FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "i.013");
  I->addIncoming(FuncBuilder.getInt32(0), EntryBB);

  // Create PHI node for source pointer
  PHINode *SrcPtrLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "src.ptr.loop");
  SrcPtrLoop->addIncoming(SrcArg, EntryBB);

  // Create PHI node for destination pointer
  PHINode *DstPtrLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "dst.ptr.loop");
  DstPtrLoop->addIncoming(DstArg, EntryBB);

  Value *Inc =
      FuncBuilder.CreateAdd(I, FuncBuilder.getInt32(1), "inc", true, true);
  I->addIncoming(Inc, ForBodyBB);

  // First group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcPtrLoop, 2, 16, 0, 1);
  DstArg = createEspVst128Ip(FuncBuilder, DstPtrLoop, 0);

  // Second group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcArg, 0, 16, 1, 2);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 1);

  // Third group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcArg, 1, 16, 2, 0);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 2);

  // Update PHI nodes at the end of the loop
  SrcPtrLoop->addIncoming(SrcArg, ForBodyBB);
  DstPtrLoop->addIncoming(DstArg, ForBodyBB);

  Value *ExitCond = FuncBuilder.CreateICmpEQ(
      Inc, FuncBuilder.getInt32(quotient), "exitcond.not");
  FuncBuilder.CreateCondBr(ExitCond, ForCondCleanupBB, ForBodyBB);

  FuncBuilder.SetInsertPoint(ForCondCleanupBB);
  Value *adjusted_src_ptr = FuncBuilder.CreateAdd(
      SrcArg, FuncBuilder.getInt32(-32), "adjusted_src_ptr");

  Value *DstPtr = FuncBuilder.CreateIntToPtr(DstArg, FuncBuilder.getPtrTy());
  Value *SrcPtr =
      FuncBuilder.CreateIntToPtr(adjusted_src_ptr, FuncBuilder.getPtrTy());

  FuncBuilder.CreateMemCpy(DstPtr, Align(1), SrcPtr, Align(1),
                           FuncBuilder.getInt32(remainder - 32));

  FuncBuilder.CreateRetVoid();

  M->eraseFromParent();
  return true;
}
// src unaligned, dst 16-byte aligned, size is divisible by 16
bool RISCVEsp32P4MemIntrinPass::processSrcUnalignDst16Const16(
    MemCpyInst *M, BasicBlock::iterator &BBI) {
  uint64_t quotient = Len / 48;
  uint64_t remainder = Len % 48;
  if (Len < 16) {
    return false; // not process, use memcpy
  }
  if (quotient == 0) {
    if (remainder >= 1 && remainder <= 15) {
      return true;
    }

    if (remainder >= 16 && remainder <= 31) {

      IRBuilder<> Builder(M);
      Value *Src = M->getSource();
      Value *Dst = M->getDest();

      Value *SrcAddr = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());
      Value *DstAddr = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());
      static int FuncCounter = 0;
      std::string FuncName = "esp32p4MemCpySrcunalignedDst16From16to31Index" +
                             std::to_string(FuncCounter++);

      Function *MemCpyFunc =
          createMemCpyHelperFunction(Builder, FuncName, DstAddr, SrcAddr);

      Value *DstArg = MemCpyFunc->arg_begin();
      DstArg->setName("dst");
      Value *SrcArg = MemCpyFunc->arg_begin() + 1;
      SrcArg->setName("src");

      BasicBlock *EntryBB =
          BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
      IRBuilder<> FuncBuilder(EntryBB);
      // remainder in [16,31]
      FuncBuilder.SetInsertPoint(EntryBB);
      SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 0);
      SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 1);

      createEspSrcQ(FuncBuilder, 0, 0, 1);
      DstArg = createEspVst128Ip(FuncBuilder, DstArg, 0);

      Value *DstPtr =
          FuncBuilder.CreateIntToPtr(DstArg, FuncBuilder.getPtrTy());
      Value *SrcPtr =
          FuncBuilder.CreateIntToPtr(SrcArg, FuncBuilder.getPtrTy());
      FuncBuilder.CreateMemCpy(DstPtr, Align(16), SrcPtr, Align(1),
                               FuncBuilder.getInt32(remainder - 16));

      FuncBuilder.CreateRetVoid();
      M->eraseFromParent();
      return true;
    } else if (remainder >= 32 && remainder <= 47) {
      IRBuilder<> Builder(M);
      Value *Src = M->getSource();
      Value *Dst = M->getDest();

      Value *SrcAddr = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());
      Value *DstAddr = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());

      std::string FuncName = "esp32p4MemCpySrcunalignedDst16From32To47";
      // Check if the function already exists in the current module
      if (Function *ExistingFunc = M->getModule()->getFunction(FuncName)) {
        // If the function exists, create a call directly
        Builder.CreateCall(ExistingFunc, {DstAddr, SrcAddr});
        M->eraseFromParent();
        return true;
      }

      // Create new function
      Function *MemCpyFunc =
          createMemCpyHelperFunction(Builder, FuncName, DstAddr, SrcAddr);
      Value *DstArg = MemCpyFunc->arg_begin();
      DstArg->setName("dst");
      Value *SrcArg = MemCpyFunc->arg_begin() + 1;
      SrcArg->setName("src");

      BasicBlock *EntryBB =
          BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
      IRBuilder<> FuncBuilder(EntryBB);
      // remainder in [16,31]
      FuncBuilder.SetInsertPoint(EntryBB);
      SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 0);
      SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 1);

      SrcArg = createEspSrcQLdIp(FuncBuilder, SrcArg, 2, 0, 0, 1);
      DstArg = createEspVst128Ip(FuncBuilder, DstArg, 0);
      createEspSrcQ(FuncBuilder, 1, 1, 2);
      DstArg = createEspVst128Ip(FuncBuilder, DstArg, 1);

      Value *DstPtr =
          FuncBuilder.CreateIntToPtr(DstArg, FuncBuilder.getPtrTy());
      Value *SrcPtr =
          FuncBuilder.CreateIntToPtr(SrcArg, FuncBuilder.getPtrTy());
      FuncBuilder.CreateMemCpy(DstPtr, Align(16), SrcPtr, Align(1),
                               FuncBuilder.getInt32(remainder - 32));

      FuncBuilder.CreateRetVoid();
      M->eraseFromParent();
      return true;
    }
    return false; // not process, use memcpy
  } else {
    if (remainder == 0) {
      return processSrcUnalignDst16ConstDiv48(M, BBI, quotient);
    } else if (remainder >= 32 && remainder <= 47) {
      return processSrcUnalignDst16ConstMod48From32To47(M, BBI, quotient,
                                                        remainder);
    } else if (remainder >= 16 && remainder <= 31) {
      return processSrcUnalignDst16ConstMod48From16To31(M, BBI, quotient,
                                                        remainder);
    } else if (remainder >= 1 && remainder <= 15) {
      return processSrcUnalignDst16ConstMod48From1To15(M, BBI, quotient,
                                                       remainder);
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

  Value *SrcAddr = Builder.CreatePtrToInt(Src, Builder.getInt32Ty());
  Value *DstAddr = Builder.CreatePtrToInt(Dst, Builder.getInt32Ty());

  std::string FuncName = "esp32p4MemCpySrcunalignedDst16Var";

  // Check if the function already exists in the current module
  if (useExistingHelperFunction(M, Builder, FuncName, DstAddr, SrcAddr, Size)) {
    return true;
  }
  // Create new function type
  Function *MemCpyFunc = createMemCpyHelperFunction(Builder, FuncName, DstAddr,
                                                    SrcAddr, Size, isInline);
  // Extract function arguments
  Value *DstArg = MemCpyFunc->arg_begin();
  DstArg->setName("dst");
  Value *DstArgOrg = DstArg;
  Value *SrcArg = MemCpyFunc->arg_begin() + 1;
  SrcArg->setName("src");
  Value *SrcArgOrg = SrcArg;
  Value *SizeArg = MemCpyFunc->arg_begin() + 2;
  SizeArg->setName("size");
  // Create entry block
  BasicBlock *EntryBB =
      BasicBlock::Create(M->getContext(), "entry", MemCpyFunc);
  IRBuilder<> FuncBuilder(EntryBB);

  BasicBlock *IfEndBB =
      BasicBlock::Create(M->getContext(), "if.end", MemCpyFunc);
  // Create basic blocks
  BasicBlock *ForCondCleanupBB =
      BasicBlock::Create(M->getContext(), "for.cond.cleanup", MemCpyFunc);
  BasicBlock *ForBodyBB =
      BasicBlock::Create(M->getContext(), "for.body", MemCpyFunc);
  BasicBlock *IfThen2BB =
      BasicBlock::Create(M->getContext(), "if.then2", MemCpyFunc);
  BasicBlock *IfEnd3BB =
      BasicBlock::Create(M->getContext(), "if.end3", MemCpyFunc);
  BasicBlock *IfThen6BB =
      BasicBlock::Create(M->getContext(), "if.then6", MemCpyFunc);
  BasicBlock *IfEnd7BB =
      BasicBlock::Create(M->getContext(), "if.end7", MemCpyFunc);
  BasicBlock *CleanupOutBB =
      BasicBlock::Create(M->getContext(), "cleanup.out", MemCpyFunc);

  BasicBlock *CallCleanupBB =
      BasicBlock::Create(M->getContext(), "call.cleanup", MemCpyFunc);

  BasicBlock *ReturnBB =
      BasicBlock::Create(M->getContext(), "return", MemCpyFunc);

  Value *Cmp =
      FuncBuilder.CreateICmpULT(SizeArg, FuncBuilder.getInt32(16), "cmp");

  FuncBuilder.CreateCondBr(Cmp, CleanupOutBB, IfEndBB);

  FuncBuilder.SetInsertPoint(IfEndBB);

  // Calculate the number of loops and the remainder
  Value *Div = FuncBuilder.CreateUDiv(SizeArg, FuncBuilder.getInt32(48), "div");
  Value *Mul = FuncBuilder.CreateMul(Div, FuncBuilder.getInt32(48));
  Value *Rem = FuncBuilder.CreateSub(SizeArg, Mul, "rem.decomposed");
  // Generate the complete SIMD instruction sequence
  SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 0);
  SrcArg = createEspLd128UsarIp(FuncBuilder, SrcArg, 1);

  // Entry block logic
  Value *Cmp21_not =
      FuncBuilder.CreateICmpULT(SizeArg, FuncBuilder.getInt32(48), "cmp21.not");
  FuncBuilder.CreateCondBr(Cmp21_not, ForCondCleanupBB, ForBodyBB);

  // Remainder processing logic
  FuncBuilder.SetInsertPoint(ForCondCleanupBB);
  // Create PHI nodes to track the source pointer, destination pointer, and
  // remaining bytes after the loop
  PHINode *SrcPtrAfterLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "src.ptr.afterloop");
  SrcPtrAfterLoop->addIncoming(SrcArg, IfEndBB);

  PHINode *DstPtrAfterLoop =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "dst.ptr.afterloop");
  DstPtrAfterLoop->addIncoming(DstArg, IfEndBB);

  // PHINode *RemAfterLoop = FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2,
  // "rem.afterloop"); RemAfterLoop->addIncoming(Rem, IfEndBB);
  Value *RemCmp =
      FuncBuilder.CreateICmpULT(Rem, FuncBuilder.getInt32(32), "tobool.not");
  FuncBuilder.CreateCondBr(RemCmp, IfEnd3BB, IfThen2BB);

  // Loop body
  FuncBuilder.SetInsertPoint(ForBodyBB);
  PHINode *I = FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "i.022");
  I->addIncoming(FuncBuilder.getInt32(0), IfEndBB);

  PHINode *SrcPtrPhi =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "src.ptr.phi");
  SrcPtrPhi->addIncoming(SrcArg, IfEndBB);

  PHINode *DstPtrPhi =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 2, "dst.ptr.phi");
  DstPtrPhi->addIncoming(DstArg, IfEndBB);

  // First group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcPtrPhi, 2, 16, 0, 1);
  DstArg = createEspVst128Ip(FuncBuilder, DstPtrPhi, 0);

  // Second group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcArg, 0, 16, 1, 2);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 1);

  // Third group of operations
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcArg, 1, 16, 2, 0);
  SrcArg->setName(SrcArg->getName() + ".final");
  SrcPtrPhi->addIncoming(SrcArg, ForBodyBB);
  DstArg = createEspVst128Ip(FuncBuilder, DstArg, 2);
  DstArg->setName(DstArg->getName() + ".final");
  DstPtrPhi->addIncoming(DstArg, ForBodyBB);
  // Loop control
  Value *Inc =
      FuncBuilder.CreateAdd(I, FuncBuilder.getInt32(1), "inc", true, true);
  I->addIncoming(Inc, ForBodyBB);
  Value *ExitCond = FuncBuilder.CreateICmpEQ(Inc, Div, "exitcond.not");
  FuncBuilder.CreateCondBr(ExitCond, ForCondCleanupBB, ForBodyBB);

  SrcPtrAfterLoop->addIncoming(SrcArg, ForBodyBB);
  DstPtrAfterLoop->addIncoming(DstArg, ForBodyBB);
  // RemAfterLoop->addIncoming(Rem, ForBodyBB);

  // Process the remainder of more than 32 bytes
  FuncBuilder.SetInsertPoint(IfThen2BB);
  SrcArg = createEspSrcQLdIp(FuncBuilder, SrcPtrAfterLoop, 2, 0, 0, 1);
  DstArg = createEspVst128Ip(FuncBuilder, DstPtrAfterLoop, 0);
  createEspSrcQ(FuncBuilder, 1, 1, 2);
  Value *DstArgIfThen2 = DstArg = createEspVst128Ip(FuncBuilder, DstArg, 1);
  Value *Sub1 = FuncBuilder.CreateAdd(Rem, FuncBuilder.getInt32(-32), "sub1",
                                      false, true);
  FuncBuilder.CreateBr(CleanupOutBB);

  // Process the remainder of 16 bytes
  FuncBuilder.SetInsertPoint(IfEnd3BB);
  Value *Cmp16 =
      FuncBuilder.CreateICmpULT(Rem, FuncBuilder.getInt32(16), "tobool5.not");
  FuncBuilder.CreateCondBr(Cmp16, IfEnd7BB, IfThen6BB);

  FuncBuilder.SetInsertPoint(IfThen6BB);
  createEspSrcQ(FuncBuilder, 0, 0, 1);
  DstArg = createEspVst128Ip(FuncBuilder, DstPtrAfterLoop, 0);
  Value *SubSrc = FuncBuilder.CreateAdd(SrcPtrAfterLoop,
                                        FuncBuilder.getInt32(-16), "sub_src");
  Value *Sub9 = FuncBuilder.CreateAdd(Rem, FuncBuilder.getInt32(-16), "sub9",
                                      false, true);
  FuncBuilder.CreateBr(CleanupOutBB);

  FuncBuilder.SetInsertPoint(IfEnd7BB);

  PHINode *sub_src_32 =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 1, "sub_src_32");
  // Create the initial value of sub_src_32
  sub_src_32->addIncoming(SrcPtrAfterLoop, IfEnd3BB);

  PHINode *DstEnd7 =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 1, "dst_end7");
  DstEnd7->addIncoming(DstPtrAfterLoop, IfEnd3BB);

  PHINode *RemEnd7 =
      FuncBuilder.CreatePHI(FuncBuilder.getInt32Ty(), 1, "rem_end7");
  RemEnd7->addIncoming(Rem, IfEnd3BB);
  // Create src_end7 = sub_src_32 - 32
  Value *SrcEnd7 =
      FuncBuilder.CreateAdd(sub_src_32, FuncBuilder.getInt32(-32), "src_end7");
  FuncBuilder.CreateBr(CleanupOutBB);

  // Final memcpy processing remaining bytes
  FuncBuilder.SetInsertPoint(CleanupOutBB);
  Value *SrcFinal =
      PHINode::Create(FuncBuilder.getInt32Ty(), 4, "src.final", CleanupOutBB);
  cast<PHINode>(SrcFinal)->addIncoming(SrcArg, IfThen2BB);
  cast<PHINode>(SrcFinal)->addIncoming(SubSrc, IfThen6BB);
  cast<PHINode>(SrcFinal)->addIncoming(SrcArgOrg, EntryBB);
  cast<PHINode>(SrcFinal)->addIncoming(SrcEnd7, IfEnd7BB);
  Value *DstFinal =
      PHINode::Create(FuncBuilder.getInt32Ty(), 4, "dst.final", CleanupOutBB);
  cast<PHINode>(DstFinal)->addIncoming(DstArgIfThen2, IfThen2BB);
  cast<PHINode>(DstFinal)->addIncoming(DstArg, IfThen6BB);
  cast<PHINode>(DstFinal)->addIncoming(DstArgOrg, EntryBB);
  cast<PHINode>(DstFinal)->addIncoming(DstEnd7, IfEnd7BB);
  Value *RemFinal =
      PHINode::Create(FuncBuilder.getInt32Ty(), 4, "rem.final", CleanupOutBB);
  cast<PHINode>(RemFinal)->addIncoming(SizeArg, EntryBB);
  cast<PHINode>(RemFinal)->addIncoming(Sub1, IfThen2BB);
  cast<PHINode>(RemFinal)->addIncoming(Sub9, IfThen6BB);
  cast<PHINode>(RemFinal)->addIncoming(RemEnd7, IfEnd7BB);
  // Check if there are any remaining bytes to process
  Value *CmpFinal =
      FuncBuilder.CreateICmpEQ(RemFinal, FuncBuilder.getInt32(0), "cmp.final");

  // Condition branch based on comparison result
  FuncBuilder.CreateCondBr(CmpFinal, ReturnBB, CallCleanupBB);

  // Set the return block
  FuncBuilder.SetInsertPoint(ReturnBB);
  FuncBuilder.CreateRetVoid();

  // Set the cleanup block, for processing the remaining bytes
  FuncBuilder.SetInsertPoint(CallCleanupBB);
  Value *DstPtrFinal = FuncBuilder.CreateIntToPtr(
      DstFinal, FuncBuilder.getPtrTy(), "dst.ptr.final");
  Value *SrcPtrFinal = FuncBuilder.CreateIntToPtr(
      SrcFinal, FuncBuilder.getPtrTy(), "src.ptr.final");
  processMemCpyVarFrom1To15(
      FuncBuilder, "esp32p4MemCpySrcUnalignDst16From1To15Opt", DstPtrFinal,
      SrcPtrFinal, RemFinal, true, MemCpyType::SrcUnalign_Dst16_Var);
  FuncBuilder.CreateBr(ReturnBB);

  M->eraseFromParent();
  return true;
}

// src unaligned, dst 16-byte aligned, size is variable
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

  // Check if the function already exists in the current module
  if (useExistingHelperFunction(M, Builder, FuncName, Dst, Src,
                                Builder.getInt32(Len))) {
    return true;
  }

  Function *MemCpyFunc = createMemCpyHelperFunctionPtr(
      Builder, FuncName, Dst, Src, Builder.getInt32(Len));
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

  // Load and store the first 8 bytes in the entry block
  Value *LoadVal =
      FuncBuilder.CreateAlignedLoad(FuncBuilder.getInt64Ty(), SrcArg, Align(1));
  FuncBuilder.CreateAlignedStore(LoadVal, DstArg, Align(1));

  // Calculate the remaining size
  Value *Sub = FuncBuilder.CreateSub(SizeArg, FuncBuilder.getInt32(8), "sub");

  Value *DstPtr = FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), DstArg,
                                        FuncBuilder.getInt32(8), "add.ptr1");
  Value *SrcPtr = FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), SrcArg,
                                        FuncBuilder.getInt32(8), "add.ptr");

  FuncBuilder.CreateMemCpy(DstPtr, Align(16), SrcPtr, Align(1), Sub);

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
  // Check if the function already exists in the current module
  if (useExistingHelperFunction(M, Builder, FuncName, Dst, Src, Size)) {
    return true;
  }

  Function *MemCpyFunc =
      createMemCpyHelperFunctionPtr(Builder, FuncName, Dst, Src, Size);
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

  // Load and store the first 8 bytes in the entry block
  Value *LoadVal =
      FuncBuilder.CreateAlignedLoad(FuncBuilder.getInt64Ty(), SrcArg, Align(1));
  FuncBuilder.CreateAlignedStore(LoadVal, DstArg, Align(1));

  Value *Cond720 = FuncBuilder.CreateSub(SizeArg, FuncBuilder.getInt32(8));
  Value *AddPtr2 = FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), DstArg,
                                         FuncBuilder.getInt32(8));
  Value *AddPtr = FuncBuilder.CreateGEP(FuncBuilder.getInt8Ty(), SrcArg,
                                        FuncBuilder.getInt32(8));

  FuncBuilder.CreateMemCpy(AddPtr2, Align(16), AddPtr, Align(1), Cond720);
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