//===- RISCVESPOperands.td - ESP32P4 instruction operands -*- tblgen-*--======//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Immediate operands with a shared generic render method.
class P4ImmAsmOperand<string name> : AsmOperandClass {
  let Name = name;
  let RenderMethod = "addImmOperands";
  let DiagnosticType = !strconcat("Invalid", name);
}

class Immediate<ValueType vt, code pred, string asmop>
  : Operand<vt>, ImmLeaf<vt, pred> {
  let PrintMethod = "print"#asmop;
  let ParserMatchClass = !cast<AsmOperandClass>(asmop);
}
/// --------------------------- ///

// imm8 predicate - Immediate in the range [-128,127]
def Imm8_AsmOperand : P4ImmAsmOperand<"Imm8">;
def imm8 : Immediate<i32, [{ return Imm >= -128 && Imm <= 127; }], "Imm8_AsmOperand"> {
  let EncoderMethod = "getImm8OpValue";
  let DecoderMethod = "decodeImm8Operand";
}

// offset_16_16 predicate - 4-bit signed immediate in the range [-128,112] with an interval
// of 16.
def Offset_16_16_AsmOperand: P4ImmAsmOperand<"Offset_16_16">;
def offset_16_16: Immediate<i32, [{ return (Imm >= -128 && Imm <= 112) && ((Imm & 0xf) == 0); }], "Offset_16_16_AsmOperand"> {
  let EncoderMethod = "getOffset_16_16OpValue";
  let DecoderMethod = "decodeOffset_16_16Operand";
}

// Offset_256_1 predicate - 8-bit signed immediate in the range [-128,127]
def offset_256_1 : RISCVSImmOp<8>;

// offset_256_8 predicate - 4-bit signed immediate in the range [-1024,1016] with an interval
// of 8.
def Offset_256_8_AsmOperand: P4ImmAsmOperand<"Offset_256_8">;
def offset_256_8: Immediate<i32, [{ return (Imm >= -1024 && Imm <= 1016) && ((Imm & 0x7) == 0); }], "Offset_256_8_AsmOperand"> {
  let EncoderMethod = "getOffset_256_8OpValue";
  let DecoderMethod = "decodeOffset_256_8Operand";
}

// offset_256_16 predicate - 8-bit signed immediate in the range [-2048,2032] with an interval
// of 16.
def Offset_256_16_AsmOperand: P4ImmAsmOperand<"Offset_256_16">;
def offset_256_16: Immediate<i32, [{ return (Imm >= -2048 && Imm <= 2032) && ((Imm & 0xf) == 0); }], "Offset_256_16_AsmOperand"> {
  let EncoderMethod = "getOffset_256_16OpValue";
  let DecoderMethod = "decodeOffset_256_16Operand";
}

// offset_256_2 predicate - 8-bit signed immediate in the range [-256,254] with an interval
// of 2.
def Offset_256_2_AsmOperand: P4ImmAsmOperand<"Offset_256_2">;
def offset_256_2: Immediate<i32, [{ return (Imm >= -256 && Imm <= 254) && ((Imm & 0x1) == 0); }], "Offset_256_2_AsmOperand"> {
  let EncoderMethod = "getOffset_256_2OpValue";
  let DecoderMethod = "decodeOffset_256_2Operand";
}

// offset_256_4 predicate - 8-bit signed immediate in the range [-512,508] with an interval
// of 4.
def Offset_256_4_AsmOperand: P4ImmAsmOperand<"Offset_256_4">;
def offset_256_4: Immediate<i32, [{ return (Imm >= -512 && Imm <= 508) && ((Imm & 0x3) == 0); }], "Offset_256_4_AsmOperand"> {
  let EncoderMethod = "getOffset_256_4OpValue";
  let DecoderMethod = "decodeOffset_256_4Operand";
}

// select_2 predicate - Immediate in the range [0,1]
def Select_2_AsmOperand: P4ImmAsmOperand<"Select_2">;
def select_2: Immediate<i32, [{ return Imm >= 0 && Imm <= 1; }], "Select_2_AsmOperand"> {
  let EncoderMethod = "getSelect_2OpValue";
  let DecoderMethod = "decodeSelect_2Operand";
}

// select_4 predicate - Immediate in the range [0,3]
def Select_4_AsmOperand: P4ImmAsmOperand<"Select_4">;
def select_4: Immediate<i32, [{ return Imm >= 0 && Imm <= 3; }], "Select_4_AsmOperand"> {
  let EncoderMethod = "getSelect_4OpValue";
  let DecoderMethod = "decodeSelect_4Operand";
}

// select_8 predicate - Immediate in the range [0,7]
def Select_8_AsmOperand: P4ImmAsmOperand<"Select_8">;
def select_8: Immediate<i32, [{ return Imm >= 0 && Imm <= 7; }], "Select_8_AsmOperand"> {
  let EncoderMethod = "getSelect_8OpValue";
  let DecoderMethod = "decodeSelect_8Operand";
}

// select_16 predicate - Immediate in the range [0,15]
def Select_16_AsmOperand: P4ImmAsmOperand<"Select_16">;
def select_16: Immediate<i32, [{ return Imm >= 0 && Imm <= 15; }], "Select_16_AsmOperand"> {
  let EncoderMethod = "getSelect_16OpValue";
  let DecoderMethod = "decodeSelect_16Operand";
}

// Espressif hardware-loop byte offsets: a bare symbol, or an even constant in
// range (encoded field is offset/2). Dedicated classes so the parser enforces
// the even/range constraint with a clean diagnostic instead of accepting odd
// values (encoder assert) or crashing on out-of-range constants; kept separate
// from the plain UImm10/UImm13 classes shared with Xqci.
def UImm10Step2AsmOperand : AsmOperandClass {
  let Name = "UImm10Step2";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidUImm10Step2";
  let PredicateMethod = "isUImm10Step2";
}
def UImm13Step2AsmOperand : AsmOperandClass {
  let Name = "UImm13Step2";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidUImm13Step2";
  let PredicateMethod = "isUImm13Step2";
}

def uimm10_step4 : Operand<OtherVT> {
  let ParserMatchClass = UImm10Step2AsmOperand;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getUImm10_Step4Operand";
  let DecoderMethod = "decodeUImm10_Step4Operand";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      // esp.lp.setupi loop offset: even byte offset in [0, 1022]
      // (must match fixup_riscv_esp_lp_offset_9 in RISCVAsmBackend.cpp).
      return isUInt<10>(Imm) && ((Imm & 1) == 0);
    return MCOp.isBareSymbolRef();
  }];
  let OperandType = "OPERAND_PCREL";
}

def uimm13_step4 : Operand<OtherVT> {
  let ParserMatchClass = UImm13Step2AsmOperand;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getUImm13_Step4Operand";
  let DecoderMethod = "decodeUImm13_Step4Operand";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      // esp.lp.setup/starti/endi loop offset: even byte offset in [0, 8190]
      // (must match fixup_riscv_esp_lp_offset_12 in RISCVAsmBackend.cpp).
      return isUInt<13>(Imm) && ((Imm & 1) == 0);
    return MCOp.isBareSymbolRef();
  }];
  let OperandType = "OPERAND_PCREL";
}

// A 12-bit unsigned immediate.
def uimm12 : RISCVOp,
                   ImmLeaf<XLenVT, [{return isUInt<12>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<12>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<12>";
  let OperandType = "OPERAND_UIMM12";
  let MCOperandPredicate = [{
    uint64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isUInt<12>(Imm);
  }];
}

def imm5 : RISCVSImmOp<5>;

def shamt : RISCVUImmOp<6> {
    let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isUInt<}]# bitsNum #[{>(Imm) && Imm <= 32>;
  }];
}

def SATAsmOperand : AsmOperandClass {
  let Name = "SATAsmOperand";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseSATArg";
  let PredicateMethod = "isSATArg";
  let IsOptional = 1;
  let DefaultMethod = "defaultSATOp";
}

def sat : RISCVUImmOp<1> {
  let ParserMatchClass = SATAsmOperand;
  let PrintMethod = "printSATOperand";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isUInt<}]# bitsNum #[{>(Imm);
  }];
}

def RMAsmOperand : AsmOperandClass {
  let Name = "RMAsmOperand";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseRMArg";
  let PredicateMethod = "isRMArg";
  let IsOptional = 1;
  let DefaultMethod = "defaultRMOp";
}

def rm : RISCVUImmOp<3> {
  let ParserMatchClass = RMAsmOperand;
  let PrintMethod = "printRMOperand";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isUInt<}]# bitsNum #[{>(Imm);
  }];
}
