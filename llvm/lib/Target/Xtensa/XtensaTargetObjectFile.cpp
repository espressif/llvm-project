//===-- llvm/Target/XtensaTargetObjectFile.cpp - Xtensa Object Info Impl --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "XtensaTargetObjectFile.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//                               ELF Target
//===----------------------------------------------------------------------===//

void XtensaElfTargetObjectFile::Initialize(MCContext &Ctx,
                                           const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(false);
}
