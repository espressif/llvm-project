//===-- RISCVESPInstrInfo.cpp - ESP Instruction Information -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements ESP/ESPV-specific TargetInstrInfo and register helper
// functions for RISC-V.
//
//===----------------------------------------------------------------------===//
#include "RISCVESPInstrInfo.h"
#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

std::optional<unsigned> llvm::getQR64LaneSubIdx(MCRegister Reg) {
  // TableGen interleaves the enums: Q0_D0, Q0_D1, Q1_D0, ..., Q7_D1.
  static_assert(RISCV::Q0_D1 == RISCV::Q0_D0 + 1 &&
                RISCV::Q7_D1 == RISCV::Q0_D0 + 15);
  if (Reg < RISCV::Q0_D0 || Reg > RISCV::Q7_D1)
    return std::nullopt;
  return (Reg - RISCV::Q0_D0) % 2 ? RISCV::sub_qr_64_hi : RISCV::sub_qr_64;
}

std::optional<unsigned>
llvm::getQR64SubRegIdxForExtractIndex(unsigned OrigIdx, unsigned VecNumElts) {
  if (OrigIdx == 0)
    return RISCV::sub_qr_64;
  if (OrigIdx == VecNumElts / 2)
    return RISCV::sub_qr_64_hi;
  return std::nullopt;
}

unsigned llvm::getQR64HiExtractIndex(unsigned VecNumElts) {
  return VecNumElts / 2;
}

bool RISCV::isESPReMaterializable(const MachineInstr &MI) {
  // ESP_ZERO_QACC and ESP_ZERO_XACC are rematerializable (side-effect free) so
  // RA rematerializes instead of spilling unspillable QACC/XACC registers.
  unsigned Opc = MI.getOpcode();
  return Opc == RISCV::ESP_ZERO_QACC || Opc == RISCV::ESP_ZERO_XACC;
}

static bool isQACCReg(Register Reg) {
  return RISCV::QACC_LRegClass.contains(Reg) ||
         RISCV::QACC_HRegClass.contains(Reg) ||
         RISCV::QACC_L_LOWRegClass.contains(Reg) ||
         RISCV::QACC_L_HIGHRegClass.contains(Reg) ||
         RISCV::QACC_H_LOWRegClass.contains(Reg) ||
         RISCV::QACC_H_HIGHRegClass.contains(Reg);
}

static bool isESPVecReg(Register Reg) {
  return RISCV::QRRegClass.contains(Reg) || RISCV::QR_64RegClass.contains(Reg);
}

bool RISCV::copyESPPhysReg(const RISCVInstrInfo &TII, const RISCVSubtarget &STI,
                           MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, const DebugLoc &DL,
                           Register DstReg, Register SrcReg, bool KillSrc,
                           bool RenamableDest, bool RenamableSrc) {
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  unsigned SrcFlags =
      getKillRegState(KillSrc) | getRenamableRegState(RenamableSrc);
  auto EmitUnary = [&](unsigned Opc) {
    BuildMI(MBB, MBBI, DL, TII.get(Opc), DstReg).addReg(SrcReg, SrcFlags);
    return true;
  };

  bool DstIsGPR = RISCV::GPRRegClass.contains(DstReg);
  bool SrcIsGPR = RISCV::GPRRegClass.contains(SrcReg);
  bool IsV2P2 = STI.useESPV2P2Instructions();

  // Handle GPR <-> FFT_BIT_WIDTH / SAR / SAR_BYTES / XACC subregister moves.
  if (SrcIsGPR) {
    if (RISCV::FFT_BIT_WIDTHRegRegClass.contains(DstReg))
      return EmitUnary(RISCV::ESP_MOVX_W_FFT_BIT_WIDTH);
    if (RISCV::SARRegRegClass.contains(DstReg))
      return EmitUnary(RISCV::ESP_MOVX_W_SAR);
    if (RISCV::SAR_BYTESRegRegClass.contains(DstReg))
      return EmitUnary(IsV2P2 ? RISCV::ESP_MOVX_W_SAR_BYTES_2P2
                              : RISCV::ESP_MOVX_W_SAR_BYTES);
    if (RISCV::XACC_HIGHRegClass.contains(DstReg))
      return EmitUnary(RISCV::ESP_MOVX_W_XACC_H);
    if (RISCV::XACC_LOWRegClass.contains(DstReg))
      return EmitUnary(RISCV::ESP_MOVX_W_XACC_L);
  }
  if (DstIsGPR) {
    if (RISCV::FFT_BIT_WIDTHRegRegClass.contains(SrcReg))
      return EmitUnary(RISCV::ESP_MOVX_R_FFT_BIT_WIDTH);
    if (RISCV::SARRegRegClass.contains(SrcReg))
      return EmitUnary(RISCV::ESP_MOVX_R_SAR);
    if (RISCV::SAR_BYTESRegRegClass.contains(SrcReg))
      return EmitUnary(IsV2P2 ? RISCV::ESP_MOVX_R_SAR_BYTES_2P2
                              : RISCV::ESP_MOVX_R_SAR_BYTES);
    if (RISCV::XACC_HIGHRegClass.contains(SrcReg))
      return EmitUnary(RISCV::ESP_MOVX_R_XACC_H);
    if (RISCV::XACC_LOWRegClass.contains(SrcReg))
      return EmitUnary(RISCV::ESP_MOVX_R_XACC_L);
  }

  // Same-class state/accumulator/QACC<->VR copies that lower to Target COPY.
  if (RISCV::SARRegRegClass.contains(DstReg, SrcReg) ||
      RISCV::SAR_BYTESRegRegClass.contains(DstReg, SrcReg) ||
      RISCV::XACCRegRegClass.contains(DstReg, SrcReg) ||
      RISCV::XACC_LOWRegClass.contains(DstReg, SrcReg) ||
      RISCV::XACC_HIGHRegClass.contains(DstReg, SrcReg) ||
      (isESPVecReg(DstReg) && isQACCReg(SrcReg)) ||
      (isQACCReg(DstReg) && isESPVecReg(SrcReg)) ||
      (isQACCReg(DstReg) && isQACCReg(SrcReg)))
    return EmitUnary(RISCV::COPY);

  // Full 128-bit QR copy: `esp.orq qd, qs, qs` (qz = qx | qy) is a bit-exact
  // side-effect-free move that needs no GPR scratch.
  if (RISCV::QRRegClass.contains(DstReg, SrcReg)) {
    if (DstReg != SrcReg)
      BuildMI(MBB, MBBI, DL, TII.get(STI.getESPOrqOpcode()), DstReg)
          .addReg(SrcReg, getRenamableRegState(RenamableSrc))
          .addReg(SrcReg, SrcFlags);
    return true;
  }

  // QR -> QR_64 copy.
  if (RISCV::QRRegClass.contains(SrcReg) &&
      RISCV::QR_64RegClass.contains(DstReg)) {
    if (auto SubIdx = getQR64LaneSubIdx(DstReg)) {
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::EXTRACT_SUBREG), DstReg)
          .addReg(SrcReg, SrcFlags)
          .addImm(*SubIdx);
      return true;
    }
  }

  // QR_64 -> QR copy (mirrors EXTRACT_SUBREG above).
  if (RISCV::QR_64RegClass.contains(SrcReg) &&
      RISCV::QRRegClass.contains(DstReg)) {
    auto SubIdx = getQR64LaneSubIdx(SrcReg);
    if (!SubIdx)
      llvm_unreachable("Unexpected QR_64 source register");
    if (TRI->getSubReg(DstReg, *SubIdx) != MCRegister(SrcReg))
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::INSERT_SUBREG), DstReg)
          .addReg(DstReg)
          .addReg(SrcReg, SrcFlags)
          .addImm(*SubIdx);
    return true;
  }

  // QR_64 -> QR_64 copy: use idempotent full-width esp.orq when same-lane and
  // the destination's other half is dead across the copy; otherwise bridge the
  // two 32-bit lanes through a free GPRPIE scratch register.
  if (RISCV::QR_64RegClass.contains(DstReg) &&
      RISCV::QR_64RegClass.contains(SrcReg)) {
    if (DstReg == SrcReg)
      return true;

    auto getQRSuperAndBaseLane = [&](Register Reg, Register &SuperReg,
                                     unsigned &BaseLane) -> bool {
      auto SubIdx = getQR64LaneSubIdx(Reg);
      if (!SubIdx)
        return false;
      SuperReg = TRI->getMatchingSuperReg(Reg, *SubIdx, &RISCV::QRRegClass);
      BaseLane = (*SubIdx == RISCV::sub_qr_64) ? 0u : 2u;
      return SuperReg.isValid();
    };

    Register DstSuper, SrcSuper;
    unsigned DstBaseLane, SrcBaseLane;
    if (!getQRSuperAndBaseLane(DstReg, DstSuper, DstBaseLane) ||
        !getQRSuperAndBaseLane(SrcReg, SrcSuper, SrcBaseLane))
      llvm_unreachable("Unexpected QR_64 register without QR super-register");

    LiveRegUnits UsedRegs(*TRI);
    UsedRegs.addLiveOuts(MBB);
    for (auto I = MBB.end(); I != MBBI;) {
      --I;
      if (I != MBBI)
        UsedRegs.stepBackward(*I);
    }

    LivePhysRegs DefinedRegs(*TRI);
    DefinedRegs.addLiveIns(MBB);
    SmallVector<std::pair<MCPhysReg, const MachineOperand *>, 2> Clobbers;
    for (auto I = MBB.begin(); I != MBBI; ++I) {
      DefinedRegs.stepForward(*I, Clobbers);
      Clobbers.clear();
    }

    Register DstOther = TRI->getSubReg(
        DstSuper, DstBaseLane ? RISCV::sub_qr_64 : RISCV::sub_qr_64_hi);
    bool DstOtherDead =
        !DefinedRegs.contains(DstOther) || UsedRegs.available(DstOther);

    if (DstBaseLane == SrcBaseLane && DstOtherDead) {
      BuildMI(MBB, MBBI, DL, TII.get(STI.getESPOrqOpcode()), DstSuper)
          .addReg(SrcSuper)
          .addReg(SrcSuper);
      return true;
    }

    if (MBBI != MBB.end())
      UsedRegs.stepBackward(*MBBI);
    Register Scratch;
    for (Register Cand :
         {RISCV::X24, RISCV::X25, RISCV::X26, RISCV::X27, RISCV::X28,
          RISCV::X29, RISCV::X30, RISCV::X31, RISCV::X8, RISCV::X9, RISCV::X10,
          RISCV::X11, RISCV::X12, RISCV::X13, RISCV::X14, RISCV::X15}) {
      if (UsedRegs.available(Cand)) {
        Scratch = Cand;
        break;
      }
    }
    if (!Scratch)
      report_fatal_error("no free GPRPIE scratch register for ESPV QR_64 copy");

    for (unsigned Lane = 0; Lane != 2; ++Lane) {
      BuildMI(MBB, MBBI, DL, TII.get(RISCV::ESP_MOVI_32_A), Scratch)
          .addReg(SrcSuper)
          .addImm(SrcBaseLane + Lane);
      BuildMI(MBB, MBBI, DL, TII.get(RISCV::ESP_MOVI_32_Q), DstSuper)
          .addReg(DstSuper, Lane == 0 && DstOtherDead && DstSuper != SrcSuper
                                ? RegState::Undef
                                : 0)
          .addReg(Scratch, RegState::Kill)
          .addImm(DstBaseLane + Lane);
    }
    return true;
  }

  return false;
}

std::optional<unsigned>
RISCV::getESPStoreRegToStackSlotOpcode(const TargetRegisterClass *RC,
                                       const RISCVSubtarget &STI) {
  if (RISCV::QRRegClass.hasSubClassEq(RC))
    return RISCV::PseudoESP_VSPILL_128;
  if (RISCV::QR_64RegClass.hasSubClassEq(RC) && STI.hasESPVTargetLowering())
    return RISCV::PseudoESP_VSPILL_64;
  if (RISCV::XACC_HIGHRegClass.hasSubClassEq(RC) ||
      RISCV::XACC_LOWRegClass.hasSubClassEq(RC))
    return RISCV::SW;
  if (RISCV::XACCRegRegClass.hasSubClassEq(RC))
    llvm_unreachable("XACC register cannot be spilled to stack");
  return std::nullopt;
}

std::optional<unsigned>
RISCV::getESPLoadRegFromStackSlotOpcode(const TargetRegisterClass *RC,
                                        const RISCVSubtarget &STI) {
  if (RISCV::QRRegClass.hasSubClassEq(RC))
    return RISCV::PseudoESP_VRELOAD_128;
  if (RISCV::QR_64RegClass.hasSubClassEq(RC) && STI.hasESPVTargetLowering())
    return RISCV::PseudoESP_VRELOAD_64;
  if (RISCV::XACC_HIGHRegClass.hasSubClassEq(RC) ||
      RISCV::XACC_LOWRegClass.hasSubClassEq(RC))
    return RISCV::LW;
  if (RISCV::XACCRegRegClass.hasSubClassEq(RC))
    llvm_unreachable("XACC register cannot be reloaded from stack");
  return std::nullopt;
}

std::optional<DestSourcePair>
RISCV::isESPCopyInstrImpl(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return std::nullopt;
  case RISCV::ESP_ORQ:
  case RISCV::ESP_ORQ_2P2:
    if (MI.getOperand(1).isReg() && MI.getOperand(2).isReg() &&
        MI.getOperand(1).getReg() == MI.getOperand(2).getReg())
      return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
    return std::nullopt;
  }
}

bool RISCV::isESPSchedulingBoundary(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::ESP_VST_128_IP:
  case RISCV::ESP_VST_L_64_IP:
  case RISCV::ESP_VST_H_64_IP:
  case RISCV::ESP_VLDBC_8_IP:
  case RISCV::ESP_ZERO_Q:
    return true;
  default:
    return false;
  }
}
