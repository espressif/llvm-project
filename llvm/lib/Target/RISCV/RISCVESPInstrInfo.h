//===-- RISCVESPInstrInfo.h - ESP Instruction Information -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares ESP/ESPV-specific TargetInstrInfo helper functions for
// RISC-V, keeping vendor-specific logic isolated from RISCVInstrInfo.cpp.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_RISCV_RISCVESPINSTRINFO_H
#define LLVM_LIB_TARGET_RISCV_RISCVESPINSTRINFO_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include <optional>

namespace llvm {

class RISCVInstrInfo;
class RISCVSubtarget;
class TargetRegisterClass;

namespace RISCV {

bool isESPReMaterializable(const MachineInstr &MI);

bool copyESPPhysReg(const RISCVInstrInfo &TII, const RISCVSubtarget &STI,
                    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                    const DebugLoc &DL, Register DstReg, Register SrcReg,
                    bool KillSrc, bool RenamableDest, bool RenamableSrc);

std::optional<unsigned>
getESPStoreRegToStackSlotOpcode(const TargetRegisterClass *RC,
                                const RISCVSubtarget &STI);

std::optional<unsigned>
getESPLoadRegFromStackSlotOpcode(const TargetRegisterClass *RC,
                                 const RISCVSubtarget &STI);

std::optional<DestSourcePair> isESPCopyInstrImpl(const MachineInstr &MI);

bool isESPSchedulingBoundary(const MachineInstr &MI);

} // namespace RISCV
} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVESPINSTRINFO_H
