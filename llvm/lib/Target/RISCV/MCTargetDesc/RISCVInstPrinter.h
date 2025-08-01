//===-- RISCVInstPrinter.h - Convert RISC-V MCInst to asm syntax --*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints a RISC-V MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVINSTPRINTER_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVINSTPRINTER_H

#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/MC/MCInstPrinter.h"

namespace llvm {

class RISCVInstPrinter : public MCInstPrinter {
public:
  RISCVInstPrinter(const MCAsmInfo &MAI, const MCInstrInfo &MII,
                   const MCRegisterInfo &MRI)
      : MCInstPrinter(MAI, MII, MRI) {}

  bool applyTargetSpecificCLOption(StringRef Opt) override;

  void printInst(const MCInst *MI, uint64_t Address, StringRef Annot,
                 const MCSubtargetInfo &STI, raw_ostream &O) override;
  void printRegName(raw_ostream &O, MCRegister Reg) override;

  void printOperand(const MCInst *MI, unsigned OpNo, const MCSubtargetInfo &STI,
                    raw_ostream &O, const char *Modifier = nullptr);
  void printBranchOperand(const MCInst *MI, uint64_t Address, unsigned OpNo,
                          const MCSubtargetInfo &STI, raw_ostream &O);
  void printCSRSystemRegister(const MCInst *MI, unsigned OpNo,
                              const MCSubtargetInfo &STI, raw_ostream &O);
  void printFenceArg(const MCInst *MI, unsigned OpNo,
                     const MCSubtargetInfo &STI, raw_ostream &O);
  void printFRMArg(const MCInst *MI, unsigned OpNo, const MCSubtargetInfo &STI,
                   raw_ostream &O);
  void printFRMArgLegacy(const MCInst *MI, unsigned OpNo,
                         const MCSubtargetInfo &STI, raw_ostream &O);
  void printFPImmOperand(const MCInst *MI, unsigned OpNo,
                         const MCSubtargetInfo &STI, raw_ostream &O);
  void printZeroOffsetMemOp(const MCInst *MI, unsigned OpNo,
                            const MCSubtargetInfo &STI, raw_ostream &O);
  void printVTypeI(const MCInst *MI, unsigned OpNo, const MCSubtargetInfo &STI,
                   raw_ostream &O);
  void printVMaskReg(const MCInst *MI, unsigned OpNo,
                     const MCSubtargetInfo &STI, raw_ostream &O);
  void printRlist(const MCInst *MI, unsigned OpNo, const MCSubtargetInfo &STI,
                  raw_ostream &O);
  void printStackAdj(const MCInst *MI, unsigned OpNo,
                     const MCSubtargetInfo &STI, raw_ostream &O,
                     bool Negate = false);
  void printNegStackAdj(const MCInst *MI, unsigned OpNo,
                        const MCSubtargetInfo &STI, raw_ostream &O) {
    return printStackAdj(MI, OpNo, STI, O, /*Negate*/ true);
  }
  void printRegReg(const MCInst *MI, unsigned OpNo, const MCSubtargetInfo &STI,
                   raw_ostream &O);
  // Autogenerated by tblgen.
  std::pair<const char *, uint64_t>
  getMnemonic(const MCInst &MI) const override;
  void printInstruction(const MCInst *MI, uint64_t Address,
                        const MCSubtargetInfo &STI, raw_ostream &O);
  bool printAliasInstr(const MCInst *MI, uint64_t Address,
                       const MCSubtargetInfo &STI, raw_ostream &O);
  void printCustomAliasOperand(const MCInst *MI, uint64_t Address,
                               unsigned OpIdx, unsigned PrintMethodIdx,
                               const MCSubtargetInfo &STI, raw_ostream &O);
  static const char *getRegisterName(MCRegister Reg);
  static const char *getRegisterName(MCRegister Reg, unsigned AltIdx);

  void printImm8_AsmOperand(const MCInst *MI, int OpNum,
                            const MCSubtargetInfo &STI, raw_ostream &O);
  void printSelect_2_AsmOperand(const MCInst *MI, int OpNum,
                                const MCSubtargetInfo &STI, raw_ostream &O);
  void printSelect_4_AsmOperand(const MCInst *MI, int OpNum,
                                const MCSubtargetInfo &STI, raw_ostream &O);
  void printSelect_8_AsmOperand(const MCInst *MI, int OpNum,
                                const MCSubtargetInfo &STI, raw_ostream &O);
  void printSelect_16_AsmOperand(const MCInst *MI, int OpNum,
                                 const MCSubtargetInfo &STI, raw_ostream &O);
  void printOffset_16_16_AsmOperand(const MCInst *MI, int OpNum,
                                    const MCSubtargetInfo &STI, raw_ostream &O);
  void printOffset_256_8_AsmOperand(const MCInst *MI, int OpNum,
                                    const MCSubtargetInfo &STI, raw_ostream &O);
  void printOffset_256_16_AsmOperand(const MCInst *MI, int OpNum,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O);
  void printOffset_256_4_AsmOperand(const MCInst *MI, int OpNum,
                                    const MCSubtargetInfo &STI, raw_ostream &O);
};
} // namespace llvm

#endif
