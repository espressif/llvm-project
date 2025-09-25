
//===- RISCVESP32H4Operands.td - ESP32P4 instruction operands -*- tblgen-*--===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def imm5 : RISCVSImmOp<5>;
def shamt : RISCVUImmOp<6> {
    let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isUInt<}]# bitsNum #[{>(Imm) && Imm <= 32>;
  }];
}