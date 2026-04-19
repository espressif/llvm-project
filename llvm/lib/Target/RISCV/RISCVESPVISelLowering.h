//===-- RISCVESPVISelLowering.h - ESPV DAG Lowering Interface -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines ESPV-specific lowering functions for RISC-V.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVESPVISELLOWERING_H
#define LLVM_LIB_TARGET_RISCV_RISCVESPVISELLOWERING_H

#include "llvm/CodeGen/SelectionDAG.h"

namespace llvm {
class RISCVSubtarget;
class RISCVTargetLowering;

namespace RISCV {

// ESPV intrinsic lowering functions
SDValue lowerESPVIntrinsicWOChain(SDValue Op, SelectionDAG &DAG,
                                  const RISCVSubtarget &Subtarget);
SDValue lowerESPVIntrinsicWChain(SDValue Op, SelectionDAG &DAG,
                                  const RISCVSubtarget &Subtarget);

} // namespace RISCV
} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVESPVISELLOWERING_H

