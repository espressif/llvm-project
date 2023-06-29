//===- XtensaInstrInfo.td - Target Description for Xtensa -*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Xtensa instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "XtensaInstrFormats.td"
include "XtensaOperands.td"
include "XtensaOperators.td"

//===----------------------------------------------------------------------===//
// Arithmetic & Logical instructions
//===----------------------------------------------------------------------===//

class ArithLogic_RRR<bits<4> oper2, bits<4> oper1, string instrAsm,
      SDPatternOperator opNode, bit isComm = 0>
  : RRR_Inst<0x00, oper1, oper2, (outs AR:$r), (ins AR:$s, AR:$t),
             instrAsm#"\t$r, $s, $t",
            [(set AR:$r, (opNode AR:$s, AR:$t))]> {
  let isCommutable = isComm;
  let isReMaterializable = 0;
}

class Arith_Pat<ValueType res_vt, RegisterClass op_vt, 
                SDPatternOperator opNode, Instruction inst>
  : Pat<(res_vt (opNode op_vt:$r, op_vt:$t)),
          (inst op_vt:$r, op_vt:$t)>; 

def ADD : ArithLogic_RRR<0x08, 0x00, "add", add, 1>;
def SUB : ArithLogic_RRR<0x0C, 0x00, "sub", sub>;
def AND : ArithLogic_RRR<0x01, 0x00, "and", and, 1>;
def OR  : ArithLogic_RRR<0x02, 0x00, "or", or, 1>;
def XOR : ArithLogic_RRR<0x03, 0x00, "xor", xor, 1>;

class ADDX<bits<4> oper, string instrAsm, list<dag> pattern>
  : RRR_Inst<0x00, 0x00, oper, (outs AR:$r), (ins AR:$s, AR:$t),
             instrAsm#"\t$r, $s, $t", pattern>;

def ADDX2 : ADDX<0x09, "addx2", [(set AR:$r, (add AR:$t, (shl AR:$s, (i32 1))))]>;
def ADDX4 : ADDX<0x0A, "addx4", [(set AR:$r, (add AR:$t, (shl AR:$s, (i32 2))))]>;
def ADDX8 : ADDX<0x0B, "addx8", [(set AR:$r, (add AR:$t, (shl AR:$s, (i32 3))))]>;

class SUBX<bits<4> oper, string instrAsm, list<dag> pattern>
  : RRR_Inst<0x00, 0x00, oper, (outs AR:$r), (ins AR:$s, AR:$t),
             instrAsm#"\t$r, $s, $t", pattern>;

def SUBX2 : SUBX<0x0D, "subx2", [(set AR:$r, (sub (shl AR:$s, (i32 1)), AR:$t))]>;
def SUBX4 : SUBX<0x0E, "subx4", [(set AR:$r, (sub (shl AR:$s, (i32 2)), AR:$t))]>;
def SUBX8 : SUBX<0x0F, "subx8", [(set AR:$r, (sub (shl AR:$s, (i32 3)), AR:$t))]>;

def ABS : RRR_Inst<0x00, 0x00, 0x06, (outs AR:$r), (ins AR:$t),
                  "abs\t$r, $t", []> {
  let s = 0x1;
}

def ADDI : RRI8_Inst<0x02, (outs AR:$t), (ins AR:$s, imm8:$imm8),
                    "addi\t$t, $s, $imm8",
                    [(set AR:$t, (add AR:$s, imm8:$imm8))]> {
  let r = 0x0C;
}

def ADDMI : RRI8_Inst<0x02, (outs AR:$t), (ins AR:$s, imm8_sh8:$imm_sh8),
                     "addmi\t$t, $s, $imm_sh8",
                     [(set AR:$t, (add AR:$s, imm8_sh8:$imm_sh8))]> {
  bits<16> imm_sh8;

  let r = 0x0D;
  let imm8 = imm_sh8{15-8};
}

def NEG : RRR_Inst<0x00, 0x00, 0x06, (outs AR:$r), (ins AR:$t),
                  "neg\t$r, $t",
                  [(set AR:$r, (ineg AR:$t))]> {
  let s = 0x00;
}

//===----------------------------------------------------------------------===//
// Move instructions
//===----------------------------------------------------------------------===//
def MOVI : RRI8_Inst<0x02, (outs AR:$t), (ins imm12m:$imm),
                    "movi\t$t, $imm",
                    [(set AR:$t, imm12m:$imm)]> {
  bits<12> imm;

  let imm8{7-0} = imm{7-0};
  let s{3-0} = imm{11-8};
  let r = 0xa;
}

def MOVEQZ : RRR_Inst<0x00, 0x03, 0x08, (outs AR:$r), (ins AR:$s, AR:$t),
                     "moveqz\t$r, $s, $t", []>;
def MOVNEZ : RRR_Inst<0x00, 0x03, 0x09, (outs AR:$r), (ins AR:$s, AR:$t),
                     "movnez\t$r, $s, $t", []>;
def MOVLTZ : RRR_Inst<0x00, 0x03, 0x0A, (outs AR:$r), (ins AR:$s, AR:$t),
                     "movltz\t$r, $s, $t", []>;
def MOVGEZ : RRR_Inst<0x00, 0x03, 0x0B, (outs AR:$r), (ins AR:$s, AR:$t),
                     "movgez\t$r, $s, $t", []>;

//===----------------------------------------------------------------------===//
// Shift instructions
//===----------------------------------------------------------------------===//

let Uses = [SAR] in {
  def SLL : RRR_Inst<0x00, 0x01, 0x0A, (outs AR:$r), (ins AR:$s),
                   "sll\t$r, $s",
                   [(set AR:$r, (Xtensa_shl AR:$s))]> {
    let t = 0x00;
  }

  def SRA : RRR_Inst<0x00, 0x01, 0x0B, (outs AR:$r), (ins AR:$t),
                    "sra\t$r, $t",
                    [(set AR:$r, (Xtensa_sra AR:$t))]> {
    let s = 0x00;
  }

  def SRC : RRR_Inst<0x00, 0x01, 0x08, (outs AR:$r), (ins AR:$s, AR:$t),
                    "src\t$r, $s, $t",
                    [(set AR:$r, (Xtensa_src AR:$s, AR:$t))]>;

  def SRL : RRR_Inst<0x00, 0x01, 0x09, (outs AR:$r), (ins AR:$t),
                    "srl\t$r, $t",
                    [(set AR:$r, (Xtensa_srl AR:$t))]> {
    let s = 0x00;
  }
}

let Defs = [SAR] in {
  def SSL : RRR_Inst<0x00, 0x00, 0x04, (outs), (ins AR:$s),
                    "ssl\t$s",
                    [(Xtensa_ssl AR:$s)]> {
    let r = 0x01;
    let t = 0x00;
  }

  def SSR : RRR_Inst<0x00, 0x00, 0x04, (outs), (ins AR:$s),
                    "ssr\t$s", 
                    [(Xtensa_ssr AR:$s)]> {
    let r = 0x00;
    let t = 0x00;
  }
}

def EXTUI : RRR_Inst<0x00, 0x04, 0x00, (outs AR:$r), (ins AR:$t, uimm5:$imm1, imm1_16:$imm2),
                    "extui\t$r, $t, $imm1, $imm2", []> {
  bits<5> imm1;
  bits<4> imm2;

  let s = imm1{3-0};
  let Inst{16} = imm1{4};
  let Inst{23-20} = imm2;
}

def SRAI : RRR_Inst<0x00, 0x01, 0x02, (outs AR:$r), (ins AR:$t, uimm5:$sa),
                   "srai\t$r, $t, $sa",
                   [(set AR:$r, (sra AR:$t, uimm5:$sa))]> {
  bits<5> sa;

  let Inst{20} = sa{4};
  let s = sa{3-0};
}

def SRLI : RRR_Inst<0x00, 0x01, 0x04, (outs AR:$r), (ins AR:$t, uimm4:$sa),
                   "srli\t$r, $t, $sa",
                   [(set AR:$r, (srl AR:$t, uimm4:$sa))]> {
  bits<4> sa;

  let s = sa;
}

def SLLI : RRR_Inst<0x00, 0x01, 0x00, (outs AR:$r), (ins AR:$s, shimm1_31:$sa),
                   "slli\t$r, $s, $sa",
                   [(set AR:$r, (shl AR:$s, shimm1_31:$sa))]> {
  bits<5> sa;

  let Inst{20} = sa{4};
  let t = sa{3-0};
}

def SSA8L : RRR_Inst<0x00, 0x00, 0x04, (outs), (ins AR:$s),
                    "ssa8l\t$s", []> {
  let r = 0x2;
  let t = 0x0;
}

def SSAI : RRR_Inst<0x00, 0x00, 0x04, (outs), (ins uimm5:$imm),
                   "ssai\t$imm", []> {
  bits<5> imm;

  let r = 0x04;
  let s = imm{3-0};
  let t{3-1} = 0;
  let t{0} = imm{4};
}

// Shift Pseudo instructions:
// SSL/SSR + Shift combination
let usesCustomInserter = 1 in {
  def SLL_P : Pseudo<(outs AR:$r), (ins AR:$s, AR:$sa),
                    "# SLL_P $r, $s, $sa",
                    [(set AR:$r, (shl AR:$s, AR:$sa))]>;
 
  def SRA_P : Pseudo<(outs AR:$r), (ins AR:$t, AR:$sa),
                    "# SRA_P $r, $t, $sa",
                    [(set AR:$r, (sra AR:$t, AR:$sa))]>;

  def SRL_P : Pseudo<(outs AR:$r), (ins AR:$t, AR:$sa),
                    "# SRL_P $r, $t, $sa",
                    [(set AR:$r, (srl AR:$t, AR:$sa))]>;
}

//===----------------------------------------------------------------------===//
// Load and store instructions
//===----------------------------------------------------------------------===//

// Load instructions
let mayLoad = 1, usesCustomInserter = 1 in {

  class Load_RRI8<bits<4> oper, string instrAsm, SDPatternOperator opNode,
        ComplexPattern addrOp, Operand memOp>
	  : RRI8_Inst<0x02, (outs AR:$t), (ins memOp:$addr),
                instrAsm#"\t$t, $addr",
               [(set AR:$t, (opNode addrOp:$addr))]> {
    bits<12> addr;

    let r = oper;
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }
}

def L8UI  : Load_RRI8<0x00, "l8ui", zextloadi8, addr_ish1, mem8>;
def L16SI : Load_RRI8<0x09, "l16si", sextloadi16, addr_ish2, mem16>;
def L16UI : Load_RRI8<0x01, "l16ui", zextloadi16, addr_ish2, mem16>;
def L32I  : Load_RRI8<0x02, "l32i", load, addr_ish4, mem32>;

// Store instructions
let mayStore = 1, usesCustomInserter = 1 in {
  class Store_II8<bits<4> oper, string instrAsm, SDPatternOperator opNode,
        ComplexPattern addrOp, Operand memOp>
	  : RRI8_Inst<0x02, (outs), (ins AR:$t, memOp:$addr),
                instrAsm#"\t$t, $addr",
               [(opNode AR:$t, addrOp:$addr)]> {
    bits<12> addr;

    let r = oper;
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }
}

def S8I  : Store_II8<0x04, "s8i", truncstorei8, addr_ish1, mem8>;
def S16I : Store_II8<0x05, "s16i", truncstorei16, addr_ish2, mem16>;
def S32I : Store_II8<0x06, "s32i", store, addr_ish4, mem32>;

def L32R : RI16_Inst<0x01, (outs AR:$t), (ins L32Rtarget:$label),
                    "l32r\t$t, $label", []> {
  bits<16> label;
  let imm16 = label;
}

//pcrel addr loading using L32R
def : Pat<(Xtensa_pcrel_wrapper tconstpool:$in), (L32R tconstpool:$in)>;

// FrameIndexes are legalized when they are operands from load/store
// instructions. The same not happens for stack address copies, so an
// add op with mem ComplexPattern is used and the stack address copy
// can be matched.
// Setting of attribute mayLoad is trick to process instruction operands
// in function XtensaRegisterInfo::eliminateFI

let isCodeGenOnly = 1, mayLoad = 1 in {

  def LEA_ADD : RRI8_Inst<0x02, (outs AR:$t), (ins mem32:$addr),
       "addi\t$t, $addr",
       [(set AR:$t, addr_ish4:$addr)]> {
    bits<12> addr;

    let r = 0x0C;
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }
}

// Xtensa missed L8I load operation, use pseudo operation
let usesCustomInserter = 1 in
def L8I_P: Pseudo<(outs AR:$t), (ins mem8:$addr),
               "!L8I_P $t, $addr",
                [(set AR:$t, (sextloadi8
				addr_ish1:$addr))]>;

//extending loads
def : Pat<(i32 (extloadi1  addr_ish1:$addr)), (L8UI addr_ish1:$addr)>;
def : Pat<(i32 (extloadi8  addr_ish1:$addr)), (L8UI addr_ish1:$addr)>;
def : Pat<(i32 (extloadi16 addr_ish2:$addr)), (L16UI addr_ish2:$addr)>;

//===----------------------------------------------------------------------===//
// Conditional branch instructions
//===----------------------------------------------------------------------===//
let isBranch = 1, isTerminator = 1 in {
  class Branch_RR<bits<4> oper, string instrAsm, CondCode CC>
      : RRI8_Inst<0x07, (outs),
                 (ins AR:$s, AR:$t, brtarget:$target),
                  instrAsm#"\t$s, $t, $target",
                 [(brcc CC, AR:$s, AR:$t,  bb:$target)]> {
    bits<8> target;

    let r = oper;
    let imm8 = target;
  }

  class Branch_RI<bits<4> oper, string instrAsm, CondCode CC>
      : RRI8_Inst<0x06, (outs),
                 (ins AR:$s, b4const:$imm, brtarget:$target),
                  instrAsm#"\t$s, $imm, $target",
                 [(brcc CC, AR:$s, b4const:$imm,  bb:$target)]> {
    bits<4> imm;
    bits<8> target;

    let t = oper;
    let r = imm;
    let imm8 = target;
  }

  class Branch_RIU<bits<4> oper, string instrAsm, CondCode CC>
    : RRI8_Inst<0x06, (outs),
               (ins AR:$s, b4constu:$imm, brtarget:$target),
                instrAsm#"\t$s, $imm, $target",
               [(brcc CC, AR:$s, b4constu:$imm,  bb:$target)]> {
    bits<4> imm;
    bits<8> target;

    let t = oper;
    let r = imm;
    let imm8 = target;
  }

  class Branch_RZ<bits<2> n, bits<2> m, string instrAsm, CondCode CC>
    : BRI12_Inst<0x06, n, m, (outs),
                (ins AR:$s, brtarget:$target),
                 instrAsm#"\t$s, $target",
                [(brcc CC, AR:$s, (i32 0),  bb:$target)]> {
    bits<12> target;

    let imm12 = target;
  }
}

def BEQ   : Branch_RR<0x01, "beq", SETEQ>;
def BNE   : Branch_RR<0x09, "bne", SETNE>;
def BGE   : Branch_RR<0x0A, "bge", SETGE>;
def BLT   : Branch_RR<0x02, "blt", SETLT>;
def BGEU  : Branch_RR<0x0B, "bgeu", SETUGE>;
def BLTU  : Branch_RR<0x03, "bltu", SETULT>;

def BEQI  : Branch_RI<0x02, "beqi", SETEQ>;
def BNEI  : Branch_RI<0x06, "bnei", SETNE>;
def BGEI  : Branch_RI<0x0E, "bgei", SETGE>;
def BLTI  : Branch_RI<0x0A, "blti", SETLT>;
def BGEUI : Branch_RIU<0x0F, "bgeui", SETUGE>;
def BLTUI : Branch_RIU<0x0B, "bltui", SETULT>;

def BEQZ  : Branch_RZ<0x01, 0x00, "beqz", SETEQ>;
def BNEZ  : Branch_RZ<0x01, 0x01, "bnez", SETNE>;
def BGEZ  : Branch_RZ<0x01, 0x03, "bgez", SETGE>;
def BLTZ  : Branch_RZ<0x01, 0x02, "bltz", SETLT>;

def BALL : RRI8_Inst<0x07, (outs),
                    (ins AR:$s, AR:$t, brtarget:$target),
                    "ball\t$s, $t, $target", []> {
  bits<8> target;

  let r = 0x04;
  let imm8 = target;
}

def BANY : RRI8_Inst<0x07, (outs),
                    (ins AR:$s, AR:$t, brtarget:$target),
                    "bany\t$s, $t, $target", []> {
  bits<8> target;

  let r = 0x08;
  let imm8 = target;
}

def BBC : RRI8_Inst<0x07, (outs),
                   (ins AR:$s, AR:$t, brtarget:$target),
                   "bbc\t$s, $t, $target", []> {
  bits<8> target;

  let r = 0x05;
  let imm8 = target;
}

def BBS : RRI8_Inst<0x07, (outs),
                   (ins AR:$s, AR:$t, brtarget:$target),
                   "bbs\t$s, $t, $target", []> {
  bits<8> target;

  let r = 0x0d;
  let imm8 = target;
}

def BNALL : RRI8_Inst<0x07, (outs),
                    (ins AR:$s, AR:$t, brtarget:$target),
                    "bnall\t$s, $t, $target", []> {
  bits<8> target;

  let r = 0x0c;
  let imm8 = target;
}

def BNONE : RRI8_Inst<0x07, (outs),
                     (ins AR:$s, AR:$t, brtarget:$target),
                     "bnone\t$s, $t, $target", []> {
  bits<8> target;

  let r = 0x00;
  let imm8 = target;
}

def BBCI : RRI8_Inst<0x07, (outs),
                    (ins AR:$s, uimm5:$imm, brtarget:$target),
                    "bbci\t$s, $imm, $target", []> {
  bits<8> target;
  bits<5> imm;

  let r{3-1} = 0x3;
  let r{0} = imm{4};
  let t{3-0} = imm{3-0};
  let imm8 = target;
}

def : InstAlias<"bbci.l\t$s, $imm, $target", (BBCI AR:$s, uimm5:$imm, brtarget:$target)>;

def BBSI : RRI8_Inst<0x07, (outs),
                    (ins AR:$s, uimm5:$imm, brtarget:$target),
                    "bbsi\t$s, $imm, $target", []> {
  bits<8> target;
  bits<5> imm;

  let r{3-1} = 0x7;
  let r{0} = imm{4};
  let t{3-0} = imm{3-0};
  let imm8 = target;
}

def : InstAlias<"bbsi.l\t$s, $imm, $target", (BBSI AR:$s, uimm5:$imm, brtarget:$target)>;

def : Pat<(brcc SETGT, AR:$s, AR:$t, bb:$target),
          (BLT AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcc SETUGT, AR:$s, AR:$t, bb:$target),
          (BLTU AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcc SETLE, AR:$s, AR:$t, bb:$target),
          (BGE AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcc SETULE, AR:$s, AR:$t, bb:$target),
          (BGEU AR:$t, AR:$s, bb:$target)>;

def : Pat<(brcond (i32 (seteq AR:$s, AR:$t)), bb:$target),
          (BEQ AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setne AR:$s, AR:$t)), bb:$target),
          (BNE AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setge AR:$s, AR:$t)), bb:$target),
          (BGE AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setle AR:$s, AR:$t)), bb:$target),
          (BLT AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setuge AR:$s, AR:$t)), bb:$target),
          (BGEU AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setult AR:$s, AR:$t)), bb:$target),
          (BLTU AR:$s, AR:$t, bb:$target)>;
def : Pat<(brcond (i32 (setgt AR:$s, AR:$t)), bb:$target),
          (BLT AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcond (i32 (setugt AR:$s, AR:$t)), bb:$target),
          (BLTU AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcond (i32 (setle AR:$s, AR:$t)), bb:$target),
          (BGE AR:$t, AR:$s, bb:$target)>;
def : Pat<(brcond (i32 (setule AR:$s, AR:$t)), bb:$target),
          (BGEU AR:$t, AR:$s, bb:$target)>;

def : Pat<(brcond AR:$s, bb:$target), (BNEZ AR:$s, bb:$target)>;

//===----------------------------------------------------------------------===//
// Call and jump instructions
//===----------------------------------------------------------------------===//

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
  def J : CALL_Inst<0x06, (outs), (ins jumptarget:$offset),
                   "j\t$offset",
                   [(br bb:$offset)]> {
    let n = 0x0;
  }

  def JX : CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins AR:$s),
                     "jx\t$s",
                     [(brind AR:$s)]> {
    let m = 0x2;
    let n = 0x2;
    let r = 0;
    let isIndirectBranch = 1;
  }
}

let isCall = 1, Defs = [A0] in {
  def CALL0 : CALL_Inst<0x05, (outs), (ins pcrel32call:$offset),
                       "call0\t$offset", []> {
    let n = 0;
  }

  def CALLX0 : CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins AR:$s),
                         "callx0\t$s", []> {
    let m = 0x3;
    let n = 0x0;
    let r = 0;
  }
}

let isReturn = 1, isTerminator = 1,
    isBarrier = 1, Uses = [A0] in {

  def RET : CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins),
                      "ret", [(Xtensa_retflag)]> {
    let m = 0x2;
    let n = 0x0;
    let s = 0;
    let r = 0;
  }
}

// Call patterns
def : Pat<(Xtensa_call (i32 tglobaladdr:$dst)),
          (CALL0 tglobaladdr:$dst)>;
def : Pat<(Xtensa_call (i32 texternalsym:$dst)),
          (CALL0 texternalsym:$dst)>;
def : Pat<(Xtensa_call AR:$dst),
          (CALLX0 AR:$dst)>;

let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1, Size = 3 in {
  def BR_JT: Pseudo<(outs), (ins AR:$s, i32imm:$jt),
                    "!br_jt_p, $s, $jt",
                    [(Xtensa_brjt AR:$s, tjumptable:$jt)]>;
}

//===----------------------------------------------------------------------===//
// Mem barrier instructions
//===----------------------------------------------------------------------===//

def MEMW :  RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                    "memw", []> {
  let r = 0x2;
  let t = 0x0c;
  let s = 0x0;
  let hasSideEffects = 1;
}

def EXTW : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                   "extw", []> {
  let r = 0x2;
  let s = 0x0;
  let t = 0xd;
}

def : Pat<(Xtensa_mem_barrier), (MEMW)>;

//===----------------------------------------------------------------------===//
// Processor control instructions
//===----------------------------------------------------------------------===//

def DSYNC : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                    "dsync", []> {
  let r = 0x2;
  let s = 0x0;
  let t = 0x3;
}

def ISYNC : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                    "isync", []> {
  let r = 0x2;
  let s = 0x0;
  let t = 0x0;
}

def RSYNC : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                    "rsync", []> {
  let r = 0x2;
  let s = 0x0;
  let t = 0x1;
}

def ESYNC : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                    "esync", []> {
  let r = 0x2;
  let s = 0x0;
  let t = 0x2;
}

def NOP : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                  "nop", []> {
  let r = 0x02;
  let s = 0x00;
  let t = 0x0f;
}

def WSR : RSR_Inst<0x00, 0x03, 0x01, (outs SR:$sr), (ins AR:$t),
                  "wsr\t$t, $sr", []>;

def RSR : RSR_Inst<0x00, 0x03, 0x00, (outs AR:$t), (ins SR:$sr),
                  "rsr\t$t, $sr", []>;

def XSR : RSR_Inst<0x00, 0x01, 0x06, (outs AR:$ard, SR:$srd), (ins AR:$t, SR:$sr),
                  "xsr\t$t, $sr", []> {
  let Constraints = "$ard = $t, $srd = $sr";
}

//===----------------------------------------------------------------------===//
// User Registers read/write instructions
//===----------------------------------------------------------------------===//

def WUR : RRR_Inst<0x00, 0x03, 0x0F, (outs UR:$ur), (ins AR:$t),
                  "wur\t$t, $ur", []> {
  bits<8> ur;

  let r = ur{7-4};
  let s = ur{3-0};
}

def RUR : RRR_Inst<0x00, 0x03, 0x0E, (outs AR:$r), (ins UR:$ur),
                  "rur\t$r, $ur", [(set AR:$r, (Xtensa_rur UR:$ur))]> {
  bits<8> ur;

  let s = ur{7-4};
  let t = ur{3-0};
}

def WUR_FCR :RRR_Inst<0x00, 0x03, 0x0F, (outs), (ins AR:$t),
                  "wur.fcr \t$t", []> {
  let r = 232{7-4};
  let s = 232{3-0};
  let Defs = [FCR];
}

def WUR_FSR :RRR_Inst<0x00, 0x03, 0x0F, (outs), (ins AR:$t),
                  "wur.fsr\t $t", []> {
  let r = 233{7-4};
  let s = 233{3-0};
  let Defs = [FSR];
}

//===----------------------------------------------------------------------===//
// External Registers read/write instructions
//===----------------------------------------------------------------------===//

def RER : RRR_Inst<0x00, 0x00, 0x04, (outs AR:$t), (ins AR:$s),
                  "rer\t$t, $s", []> {
  let r = 0x6;
}

def WER : RRR_Inst<0x00, 0x00, 0x04, (outs), (ins AR:$t, AR:$s),
                  "wer\t$t, $s", []> {
  let r = 0x7;
  let hasSideEffects = 1;
}

//===----------------------------------------------------------------------===//
// Stack allocation
//===----------------------------------------------------------------------===//

// ADJCALLSTACKDOWN/UP implicitly use/def SP because they may be expanded into
// a stack adjustment and the codegen must know that they may modify the stack
// pointer before prolog-epilog rewriting occurs.
let Defs = [SP], Uses = [SP] in {
  def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                               "#ADJCALLSTACKDOWN",
                               [(Xtensa_callseq_start timm:$amt1, timm:$amt2)]>;
  def ADJCALLSTACKUP   : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                               "#ADJCALLSTACKUP",
                               [(Xtensa_callseq_end timm:$amt1, timm:$amt2)]>;
}

//===----------------------------------------------------------------------===//
// Generic select instruction
//===----------------------------------------------------------------------===//
let usesCustomInserter = 1 in {
  def SELECT : Pseudo<(outs AR:$dst), (ins AR:$lhs, AR:$rhs, AR:$t, AR:$f, i32imm:$cond),
                     "!select $dst, $lhs, $rhs, $t, $f, $cond",
                     [(set AR:$dst, (Xtensa_select_cc AR:$lhs, AR:$rhs, AR:$t, AR:$f, imm:$cond))]>;
}

//===----------------------------------------------------------------------===//
// Code Density instructions
//===----------------------------------------------------------------------===//

class ArithLogic_RRRN<bits<4> oper0, string instrAsm, 
      SDPatternOperator opNode, bit isComm = 0>
  : RRRN_Inst<oper0, (outs AR:$r), (ins AR:$s, AR:$t),
              instrAsm#"\t$r, $s, $t",
             [(set AR:$r, (opNode AR:$s, AR:$t))]>, Requires<[HasDensity]> {
  let isCommutable = isComm;
  let isReMaterializable = 0;
}

def ADD_N : ArithLogic_RRRN<0x0a, "add.n", add, 1>;

def ADDI_N : RRRN_Inst<0x0B, (outs AR:$r), (ins AR:$s, imm1n_15:$imm),
                      "addi.n\t$r, $s, $imm",
                      [(set AR:$r, (add AR:$s, imm1n_15:$imm))]>, Requires<[HasDensity]> {
  bits<4> imm;

  let t = imm;
}

def MOV_N : RRRN_Inst<0x0D, (outs AR:$t), (ins AR:$s),
                     "mov.n\t$t, $s", []>, Requires<[HasDensity]> {
  let r = 0;
}

def : InstAlias<"mov\t $t, $s", (OR AR:$t, AR:$s, AR:$s)>;

def MOVI_N : RI7_Inst<0xc, 0x0, (outs AR:$s), (ins imm32n_95:$imm7),
                     "movi.n\t$s, $imm7",
                     [(set AR:$s, imm32n_95:$imm7)]>, Requires<[HasDensity]>;

// Load instruction
let mayLoad = 1, usesCustomInserter = 1 in {
  def L32I_N : RRRN_Inst<0x8, (outs AR:$t), (ins mem32n:$addr), 
                        "l32i.n\t$t, $addr", []>, Requires<[HasDensity]> {
    bits<8> addr;

    let r{3-0} = addr{7-4};
    let s{3-0} = addr{3-0};
  }
}

// Store instruction
let mayStore = 1, usesCustomInserter = 1 in {
  def S32I_N : RRRN_Inst<0x9, (outs), (ins  AR:$t, mem32n:$addr),
                        "s32i.n\t$t, $addr", []>, Requires<[HasDensity]> {
    bits<8> addr;

    let r{3-0} = addr{7-4};
    let s{3-0} = addr{3-0};
  }
}

//Return instruction
let isReturn = 1, isTerminator = 1,
    isBarrier = 1, Uses = [A0] in {
  def RET_N : RRRN_Inst<0x0D, (outs), (ins),
                       "ret.n", [(Xtensa_retflag)]>,
                       Requires<[HasDensity]> {
    let r = 0x0F;
    let s = 0;
    let t = 0;
  }
}

//===----------------------------------------------------------------------===//
// Windowed instructions
//===----------------------------------------------------------------------===//

def ENTRY : BRI12_Inst<0x06, 0x3, 0x0, (outs), (ins AR:$s, entry_imm12:$imm), 
                     "entry\t$s, $imm", []>, Requires<[HasWindowed]> {
  bits<15> imm;

  let imm12{11-0} = imm{14-3};
  let Defs = [SP];
}

//Call instructions
let isCall = 1, Defs = [A0] in {
  def CALL4 : CALL_Inst<0x05, (outs), (ins pcrel32call:$offset),
                      "call4\t$offset", []>, Requires<[HasWindowed]> {
    let n = 1;
  }

  def CALL8 : CALL_Inst<0x05, (outs), (ins pcrel32call:$offset),
                       "call8\t$offset", []>, Requires<[HasWindowed]> {
    let n = 2;
  }

  def CALL12 : CALL_Inst<0x05, (outs), (ins pcrel32call:$offset),
                        "call12\t$offset", []>, Requires<[HasWindowed]> {
    let n = 3;
  }

  def CALLX4 : CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins AR:$s), 
                         "callx4\t$s", []>, Requires<[HasWindowed]> {
    let m = 0x3;
    let n = 0x1;
    let r = 0;
  }

  def CALLX8 : CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins AR:$s), 
                         "callx8\t$s", []>, Requires<[HasWindowed]> {
    let m = 0x3;
    let n = 0x2;
    let r = 0;
  }

  def CALLX12 : CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins AR:$s), 
                          "callx12\t$s", []>, Requires<[HasWindowed]> {
    let m = 0x3;
    let n = 0x3;
    let r = 0;
  }
}

//Windowed call patterns
def : Pat<(Xtensa_callw (i32 tglobaladdr:$dst)),
          (CALL8 tglobaladdr:$dst)>;
def : Pat<(Xtensa_callw (i32 texternalsym:$dst)),
          (CALL8 texternalsym:$dst)>;
def : Pat<(Xtensa_callw AR:$dst),
          (CALLX8 AR:$dst)>;

def MOVSP : RRR_Inst<0x00, 0x00, 0x00, (outs AR:$t), (ins AR:$s),
                    "movsp\t$t, $s",
                    [(set AR:$t, (Xtensa_movsp AR:$s))]>,
                    Requires<[HasWindowed]> {
  let r = 0x01;
}

//Return instructions
let isReturn = 1, isTerminator = 1,
    isBarrier = 1, Uses = [A0] in {
  def RETW_N : RRRN_Inst<0x0D, (outs), (ins),
                        "retw.n", [(Xtensa_retWflag)]>,
                        Requires<[HasWindowed, HasDensity]> {
    let r = 0x0F;
    let s = 0;
    let t = 1;
  }

  def RETW : CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins),
                       "retw", [(Xtensa_retWflag)]>,
                       Requires<[HasWindowed]> {
    let m = 0x2;
    let n = 0x1;
    let s = 0;
    let r = 0;
  }
}

//Store 32-bit for Window Exceptions
def S32E : RRI4_Inst<0x00, 0x09, (outs), (ins AR:$t, AR:$s, imm64n_4n:$imm),
                    "s32e\t$t, $s, $imm", []>, Requires<[HasWindowed]> {
  bits<6> imm;

  let r = imm{5-2};
  let imm4 = 0x4;
  let mayStore = 1;
}

def L32E : RRI4_Inst<0x00, 0x09, (outs), (ins AR:$t, AR:$s, imm64n_4n:$imm),
                    "l32e\t$t, $s, $imm", []>, Requires<[HasWindowed]> {
  bits<6> imm;

  let r = imm{5-2};
  let imm4 = 0x0;
  let mayLoad = 1;
}

//Return from window
def RFWU : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                   "rfwu", []>, Requires<[HasWindowed]> {
  bits<4> imm;

  let r = 0x3;
  let s = 0x5;
  let t = 0x0;
}

def RFWO : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                   "rfwo", []>, Requires<[HasWindowed]> {
  bits<4> imm;

  let r = 0x3;
  let s = 0x4;
  let t = 0x0;
}

//Rotate window
def ROTW : RRR_Inst<0x00, 0x00, 0x04, (outs), (ins imm8n_7:$imm),
                   "rotw\t$imm", []>, Requires<[HasWindowed]> {
  bits<4> imm;

  let r = 0x8;
  let s = 0x0;
  let t = imm{3-0};
}

//===----------------------------------------------------------------------===//
// Boolean Instructions
//===----------------------------------------------------------------------===//
class BIN_PAT<SDPatternOperator node, Instruction inst,
  ValueType src_vt, ValueType dst_vt = src_vt> 
  : Pat<(dst_vt (node src_vt:$f1, src_vt:$f2)),
        (inst src_vt:$f1, src_vt:$f2)>;


def ALL4 : RRR_Inst<0x00, 0x00, 0x00, (outs BR:$t), (ins BR:$s),
                   "all4\t$t, $s", []>, Requires<[HasBoolean]> {
  let r = 0x9;
}

def ALL8 : RRR_Inst<0x00, 0x00, 0x00, (outs BR:$t), (ins BR:$s),
                   "all8\t$t, $s", []>, Requires<[HasBoolean]> {
  let r = 0xB;
}

def ANDB : RRR_Inst<0x00, 0x02, 0x00, (outs BR:$r), (ins BR:$s, BR:$t),
                   "andb\t$r, $s, $t", []>, Requires<[HasBoolean]>;
def ANDBC : RRR_Inst<0x00, 0x02, 0x01, (outs BR:$r), (ins BR:$s, BR:$t),
                    "andbc\t$r, $s, $t", []>, Requires<[HasBoolean]>;
def ORB : RRR_Inst<0x00, 0x02, 0x02, (outs BR:$r), (ins BR:$s, BR:$t),
                  "orb\t$r, $s, $t", []>, Requires<[HasBoolean]>;
def ORBC : RRR_Inst<0x00, 0x02, 0x03, (outs BR:$r), (ins BR:$s, BR:$t),
                   "orbc\t$r, $s, $t", []>, Requires<[HasBoolean]>;
def XORB : RRR_Inst<0x00, 0x02, 0x04, (outs BR:$r), (ins BR:$s, BR:$t),
                   "xorb\t$r, $s, $t", []>, Requires<[HasBoolean]>;

def ANY4 : RRR_Inst<0x00, 0x00, 0x00, (outs BR:$t), (ins BR:$s),
                   "any4\t$t, $s", []>, Requires<[HasBoolean]> {
  let r = 0x8;
}

def ANY8 : RRR_Inst<0x00, 0x00, 0x00, (outs BR:$t), (ins BR:$s),
                   "any8\t$t, $s", []>, Requires<[HasBoolean]> {
  let r = 0xA;
}

let isBranch = 1, isTerminator = 1, Predicates = [HasBoolean] in {
  def BT : RRI8_Inst<0x06, (outs), (ins BR:$b, brtarget:$target), 
                    "bt\t$b, $target", []> {
    bits<8> target;
    bits<4> b;

    let r = 0x1;
    let s = b;
    let t = 0x7; 
    let imm8 = target;
  }

  def BF : RRI8_Inst<0x06, (outs), (ins BR:$b, brtarget:$target), 
                    "bf\t$b, $target", []> {
    bits<8> target;
    bits<4> b;

    let r = 0x0;
    let s = b;
    let t = 0x7; 
    let imm8 = target;
  }
}

let Constraints = "$dr = $r,@earlyclobber $dr" in {
  def MOVF : RRR_Inst<0x00, 0x03, 0x0C, (outs AR:$dr), (ins AR:$r, AR:$s, BR:$t),
                   "movf\t$r, $s, $t", []>, Requires<[HasBoolean]>;

  def MOVT : RRR_Inst<0x00, 0x03, 0x0D, (outs AR:$dr), (ins AR:$r, AR:$s, BR:$t),
                   "movt\t$r, $s, $t", []>, Requires<[HasBoolean]>;
}

def : Pat<(Xtensa_br_t BR:$b, bb:$target), (BT BR:$b, bb:$target)>;
def : Pat<(Xtensa_br_f BR:$b, bb:$target), (BF BR:$b, bb:$target)>;

let Predicates = [HasBoolean] in {

  def OR_BR_PAT: BIN_PAT<or,ORB,v1i1>;
  def XOR_BR_PAT: BIN_PAT<xor,XORB,v1i1>;
  def AND_BR_PAT: BIN_PAT<and,ANDB,v1i1>;

  // vselect C T F = C * T + ~C * F
  def : Pat<(v1i1 (vselect v1i1:$c, v1i1:$t, v1i1:$f)),
          (ORB (ANDB $t, $f), (ANDBC $f, $c))>;

  
  def MOVBA_P2: Pseudo<(outs BR:$r, AR:$x, AR:$y), (ins AR:$s),
                        "!movba $r, $x, $y, $s", []> {
    let Defs = [BREG];
  }
   
  def MOVBA_P: Pseudo<(outs BR:$r), (ins AR:$s),
                        "!movba $r, $s", []> {
    let usesCustomInserter = 1;
    let Defs = [BREG];
  }

  def MOVBA2_P2: Pseudo<(outs BR2:$r, AR:$x, AR:$y), (ins AR:$s),
                        "!movba $r, $x, $y, $s", []> {
    let Defs = [BREG];
  }
   
  def MOVBA2_P: Pseudo<(outs BR2:$r), (ins AR:$s),
                        "!movba2 $r, $s", []> {
    let usesCustomInserter = 1;
    let Defs = [BREG];
  }
    
  def MOVBA4_P2: Pseudo<(outs BR4:$r, AR:$x, AR:$y), (ins AR:$s),
                        "!movba4 $r, $x, $y, $s", []> {
    let Defs = [BREG];
  }
   
  def MOVBA4_P: Pseudo<(outs BR4:$r), (ins AR:$s),
                        "!movba4 $r, $s", []> {
    let usesCustomInserter = 1;
    let Defs = [BREG];
  }
  
  def EXTUI_BR_P: Pseudo<(outs AR:$r), (ins AR:$s, BR:$b),
                        "!extui_br $r, $s, $b", []>;
  def EXTUI_BR2_P: Pseudo<(outs AR:$r), (ins AR:$s, BR2:$b),
                        "!extui_br2 $r, $s, $b", []>;
  def EXTUI_BR4_P: Pseudo<(outs AR:$r), (ins AR:$s, BR4:$b),
                        "!extui_br4 $r, $s, $b", []>;
  def SLLI_BR_P: Pseudo<(outs AR:$r), (ins AR:$s, BR:$b),
                        "!slli_br $r, $s, $b", []>;

  def : Pat<(v1i1 (build_vector AR:$a)), (MOVBA_P AR:$a)>;
  def : Pat<(v1i1 (scalar_to_vector AR:$a)), (MOVBA_P AR:$a)>;
  
  def : Pat<(v2i1 (build_vector AR:$a, AR:$b)), 
             (MOVBA2_P (OR AR:$a, (SLLI AR:$b, (i32 1))))>;
  
  def : Pat<(v2i1 (Xtensa_trunc AR:$s)), (MOVBA2_P AR:$s)>;
  def : Pat<(v4i1 (Xtensa_trunc AR:$s)), (MOVBA4_P AR:$s)>;
  
  def : Pat<(i32 (vector_extract (v1i1 BR:$b), (i32 0))),
            (EXTUI_BR_P (RSR BREG), BR:$b)>;
  def : Pat<(v1i1 (load addr_ish1:$addr)), (MOVBA_P (L8UI mem8:$addr))>;
  def : Pat<(v2i1 (load addr_ish1:$addr)), (MOVBA2_P (L8UI mem8:$addr))>;
  def : Pat<(v4i1 (load addr_ish1:$addr)), (MOVBA4_P (L8UI mem8:$addr))>;

  def : Pat<(store BR:$b, addr_ish1:$addr), (S8I (EXTUI_BR_P (RSR BREG), BR:$b), mem32:$addr)>;
  def : Pat<(store BR2:$b, addr_ish1:$addr), (S8I (EXTUI_BR2_P (RSR BREG), BR2:$b), mem32:$addr)>;
  def : Pat<(store BR4:$b, addr_ish1:$addr), (S8I (EXTUI_BR4_P (RSR BREG), BR4:$b), mem32:$addr)>;

  def SPILL_BOOL: Pseudo<(outs), (ins BR:$b, mem8:$mem),
                            "!spill_bool $b, $mem",[]> {
    let mayStore = 1;
  }

  def RESTORE_BOOL: Pseudo<(outs BR:$out), (ins mem8:$mem),
                            "!restore_bool $out, $mem",[]> {
    let mayLoad = 1;
    let Defs = [BREG];
  }
}


//===----------------------------------------------------------------------===//
// Floating-Point Instructions
//===----------------------------------------------------------------------===//

class FPArith_RRR<bits<4> oper2, bits<4> oper1, string instrAsm,
                 SDPatternOperator opNode, bit isComm = 0>
  : RRR_Inst<0x00, oper1, oper2, (outs FPR:$r), (ins FPR:$s, FPR:$t),
             instrAsm#"\t$r, $s, $t",
            [(set FPR:$r, (opNode FPR:$s, FPR:$t))]> {
  let isCommutable = isComm;
  let isReMaterializable = 0;
  let Predicates = [HasSingleFloat];
}

def ADD_S : FPArith_RRR<0x00, 0x0A, "add.s", fadd, 1>;
def SUB_S : FPArith_RRR<0x01, 0x0A, "sub.s", fsub>;
def MUL_S : FPArith_RRR<0x02, 0x0A, "mul.s", fmul, 1>;

let Predicates = [HasSingleFloat] in {
  def : Arith_Pat<f32, FPR, int_xtensa_xt_add_s, ADD_S>;
  def : Arith_Pat<f32, FPR, int_xtensa_xt_sub_s, SUB_S>;
  def : Arith_Pat<f32, FPR, int_xtensa_xt_mul_s, MUL_S>;
}

// FP load instructions
let mayLoad = 1, usesCustomInserter = 1, Predicates = [HasSingleFloat] in {
  def LSI : RRI8_Inst<0x03, (outs FPR:$t), (ins mem32:$addr),
                     "lsi\t$t, $addr", []> {
    bits<12> addr;

    let r = 0x00;
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }

  def LSIP : RRI8_Inst<0x03, (outs FPR:$t, AR:$ds), (ins AR:$s, offset8m32:$off),
                      "lsip\t$t, $s, $off", []> {
    bits<8> off;
    bits<4> s;

    let r = 0x08;
    let imm8{7-0} = off{7-0};
    let s{3-0} = s{3-0};
    let Constraints = "$ds = $s,@earlyclobber $ds";
  }

  def LSX : RRR_Inst<0x00, 0x08, 0x00, (outs FPR:$r), (ins AR:$s, AR:$t),
                    "lsx\t$r, $s, $t", []>;

  def LSXP : RRR_Inst<0x00, 0x08, 0x01, (outs FPR:$r, AR:$ds), (ins AR:$s, AR:$t),
                     "lsxp\t$r, $s, $t", []> {
    let Constraints = "$ds = $s,@earlyclobber $ds";
  }
}

def : Pat<(f32 (load addr_ish4:$addr)), (f32 (LSI mem32:$addr))>;

// FP store instructions
let mayStore = 1, usesCustomInserter = 1, Predicates = [HasSingleFloat] in {
  def SSI : RRI8_Inst<0x03, (outs), (ins FPR:$t, mem32:$addr),
                     "ssi\t$t, $addr", []> {
    bits<12> addr;

    let r = 0x04;
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }

  def SSIP : RRI8_Inst<0x03, (outs AR:$ds), (ins FPR:$t, AR:$s, offset8m32:$imm),
                      "ssip\t$t, $s, $imm", []> {
    bits<12> addr;

    let r = 0x0C;
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
    let Constraints = "$ds = $s,@earlyclobber $ds";
  }

  def SSX: RRR_Inst<0x00, 0x08, 0x04, (outs), (ins FPR:$r, AR:$s, AR:$t),
                   "ssx\t$r, $s, $t", []>;

  def SSXP: RRR_Inst<0x00, 0x08, 0x05, (outs AR:$ds), (ins FPR:$r, AR:$s, AR:$t),
                    "ssxp\t$r, $s, $t", []> {
    let Constraints = "$ds = $s,@earlyclobber $ds";
  }
}

def : Pat<(store FPR:$t, addr_ish4:$addr), (SSI FPR:$t, mem32:$addr)>;

// FP compare instructions
let isCompare = 1, Predicates = [HasSingleFloat] in {
  class FCompare <bits<4> oper2, bits<4> oper1, string instrAsm,
                 SDPatternOperator opNode, bit isComm = 0>
    : RRR_Inst<0x00, oper1, oper2, (outs BR:$b), (ins FPR:$s, FPR:$t),
               instrAsm#"\t$b, $s, $t",
              [(set BR:$b, (opNode FPR:$s, FPR:$t))]> {
    let isCommutable = isComm;
    let isReMaterializable = 0;
    let Predicates = [HasSingleFloat];
  }
}

def OEQ_S :  FCompare<0x02, 0x0b, "oeq.s", Xtensa_cmpoeq, 1>;
def OLT_S :  FCompare<0x04, 0x0b, "olt.s", Xtensa_cmpolt, 0>;
def OLE_S :  FCompare<0x06, 0x0b, "ole.s", Xtensa_cmpole, 0>;

def UEQ_S :  FCompare<0x03, 0x0b, "ueq.s", Xtensa_cmpueq, 1>;
def ULT_S :  FCompare<0x05, 0x0b, "ult.s", Xtensa_cmpult, 0>;
def ULE_S :  FCompare<0x07, 0x0b, "ule.s", Xtensa_cmpule, 0>;
def UN_S  :  FCompare<0x01, 0x0b, "un.s",  Xtensa_cmpuo, 1>;

let Predicates = [HasSingleFloat] in {
  def : Arith_Pat<v1i1, FPR, int_xtensa_xt_ole_s, OLE_S>;
  def : Arith_Pat<v1i1, FPR, int_xtensa_xt_ule_s, ULE_S>;
  def : Arith_Pat<v1i1, FPR, int_xtensa_xt_olt_s, OLT_S>;
  def : Arith_Pat<v1i1, FPR, int_xtensa_xt_ult_s, ULT_S>;
  def : Arith_Pat<v1i1, FPR, int_xtensa_xt_oeq_s, OEQ_S>;
  def : Arith_Pat<v1i1, FPR, int_xtensa_xt_ueq_s, UEQ_S>;
  def : Arith_Pat<v1i1, FPR, int_xtensa_xt_un_s, UN_S>;
}

def ABS_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
                    "abs.s\t$r, $s",
                    [(set FPR:$r, (fabs FPR:$s))]>, Requires<[HasSingleFloat]> {
  let t = 0x01;
}

def : Pat<(fabs FPR:$s), (ABS_S $s)>;

def : Pat<(f32 (int_xtensa_xt_abs_s FPR:$frs)),
          (ABS_S FPR:$frs)>,
          Requires<[HasSingleFloat]>;

let Constraints = "$dr = $r,@earlyclobber $dr" in {

def ADDEXP_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$dr), (ins FPR:$r, FPR:$s),
                       "addexp.s\t$r, $s", []>, Requires<[HasSingleFloat]> {
  let t = 0x0E;
}

def ADDEXPM_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$dr), (ins FPR:$r, FPR:$s),
                        "addexpm.s\t$r, $s", []>, Requires<[HasSingleFloat]> {
  let t = 0x0F;
}
}

def CEIL_S : RRR_Inst<0x00, 0x0A, 0x0B, (outs AR:$r), (ins FPR:$s, uimm4:$imm),
                     "ceil.s\t$r, $s, $imm", []>, Requires<[HasSingleFloat]> {
  bits<4> imm;

  let t = imm;
}

def CONST_S : RRR_Inst<0x00, 0x0a, 0x0f, (outs FPR:$r), (ins uimm4:$imm),
                      "const.s\t$r, $imm", []>, Requires<[HasSingleFloat]> {
  bits<4> imm;

  let t = 0x03;
  let s = imm{3-0};
}

def DIV0_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
                     "div0.s\t$r, $s", []>, Requires<[HasSingleFloat]> {
  let t = 0x7;
}

def DIVN_S : RRR_Inst<0x00, 0x0A, 0x07, (outs FPR:$dr), (ins FPR:$r, FPR:$s, FPR:$t),
                     "divn.s\t$r, $s, $t", []>, Requires<[HasSingleFloat]> {
let Constraints = "$dr = $r,@earlyclobber $dr";
}

def FLOAT_S : RRR_Inst<0x00, 0x0A, 0x0c, (outs FPR:$r), (ins AR:$s, uimm4:$imm),
                      "float.s\t$r, $s, $imm", []>, Requires<[HasSingleFloat]> {
  bits<4> imm;

  let t = imm;
}

def : Pat<(f32 (sint_to_fp AR:$s)), (FLOAT_S AR:$s, 0)>;

def : Pat<(f32 (int_xtensa_xt_float_s i32:$s, timm:$imm)),
          (FLOAT_S AR:$s, uimm4:$imm)>,
          Requires<[HasSingleFloat]>;

def FLOOR_S : RRR_Inst<0x00, 0x0A, 0x0A, (outs AR:$r), (ins FPR:$s, uimm4:$imm),
                      "floor.s\t$r, $s, $imm", []>, Requires<[HasSingleFloat]> {
  bits<4> imm;

  let t = imm;
}

def MADDN_S : RRR_Inst<0x00, 0x0A, 0x06, (outs FPR:$dr), (ins FPR:$r, FPR:$s, FPR:$t),
                      "maddn.s\t$r, $s, $t", []>, Requires<[HasSingleFloat]> {
  let isCommutable = 0;
  let Constraints = "$dr = $r,@earlyclobber $dr";
}

// FP multipy-add
def MADD_S : RRR_Inst<0x00, 0x0A, 0x04, (outs FPR:$r), (ins FPR:$a, FPR:$s, FPR:$t),
                     "madd.s\t$r, $s, $t",
                     [(set FPR:$r, (Xtensa_madd FPR:$a, FPR:$s, FPR:$t))]>, 
                     Requires<[HasSingleFloat]> {
  let isCommutable = 0;
  let isReMaterializable = 0;
  let Constraints = "$r = $a";
}

// fmadd: r1 * r2 + r3
def : Pat<(fma FPR:$r1, FPR:$r2, FPR:$r3),
          (MADD_S $r3, $r1, $r2)>;

def MKDADJ_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$dr), (ins FPR:$r, FPR:$s),
                       "mkdadj.s\t$r, $s", []>, Requires<[HasSingleFloat]> {
  let t = 0x0D;
  let Constraints = "$dr = $r,@earlyclobber $dr";
}

def MKSADJ_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
                       "mksadj.s\t$r, $s", []>, Requires<[HasSingleFloat]> {
  let t = 0x0C;
}

// FP move instructions
def MOV_S : RRR_Inst<0x00, 0x0A, 0x0f, (outs FPR:$r), (ins FPR:$s),
                    "mov.s\t$r, $s",
                    [(set FPR:$r, (Xtensa_movs FPR:$s))]>, Requires<[HasSingleFloat]> {
  let t = 0x00;
}

let Constraints = "$dr = $r,@earlyclobber $dr" in {

  def MOVEQZ_S : RRR_Inst<0x00, 0x0B, 0x08, (outs FPR:$dr), (ins FPR:$r, FPR:$s, AR:$t),
                        "moveqz.s\t$r, $s, $t", []>, Requires<[HasSingleFloat]>;

  def MOVGEZ_S : RRR_Inst<0x00, 0x0B, 0x0B, (outs FPR:$dr), (ins FPR:$r, FPR:$s, AR:$t),
                        "movgez.s\t$r, $s, $t", []>, Requires<[HasSingleFloat]>;

  def MOVLTZ_S : RRR_Inst<0x00, 0x0B, 0x0A, (outs FPR:$dr), (ins FPR:$r, FPR:$s, AR:$t),
                        "movltz.s\t$r, $s, $t", []>, Requires<[HasSingleFloat]>;

  def MOVNEZ_S : RRR_Inst<0x00, 0x0B, 0x09, (outs FPR:$dr), (ins FPR:$r, FPR:$s, AR:$t),
                        "movnez.s\t$r, $s, $t", []>, Requires<[HasSingleFloat]>;

  def MOVT_S   : RRR_Inst<0x00, 0x0B, 0x0D, (outs FPR:$dr), (ins FPR:$r, FPR:$s, BR:$t),
                        "movt.s\t$r, $s, $t",
                        [(set FPR:$dr, (int_xtensa_xt_movt_s FPR:$r, FPR:$s, BR:$t))]>,
                        Requires<[HasBoolean, HasSingleFloat]>;

  def MOVF_S   : RRR_Inst<0x00, 0x0B, 0x0C, (outs FPR:$dr), (ins FPR:$r, FPR:$s, BR:$t),
                        "movf.s\t$r, $s, $t",
                        [(set FPR:$dr, (int_xtensa_xt_movf_s FPR:$r, FPR:$s, BR:$t))]>,
                        Requires<[HasBoolean, HasSingleFloat]>;
}

// FP multipy-sub
def MSUB_S : RRR_Inst<0x00, 0x0A, 0x05, (outs FPR:$r), (ins FPR:$a, FPR:$s, FPR:$t),
                     "msub.s\t$r, $s, $t",
                     [(set FPR:$r, (Xtensa_msub FPR:$a, FPR:$s, FPR:$t))]>, Requires<[HasSingleFloat]> {
  let isCommutable = 0;
  let isReMaterializable = 0;
  let Constraints = "$r = $a";
}

def NEXP01_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
                       "nexp01.s\t$r, $s", []>, Requires<[HasSingleFloat]> {
  let t = 0x0B;
}

def NEG_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
                    "neg.s\t$r, $s",
                    [(set FPR:$r, (fneg FPR:$s))]>, Requires<[HasSingleFloat]> {
  let t = 0x06;
}

def RECIP0_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
                       "recip0.s\t$r, $s", []>, Requires<[HasSingleFloat]> {
  let t = 0x08;
}

def RFR : RRR_Inst<0x00, 0x0A, 0x0f, (outs AR:$r), (ins FPR:$s),
                  "rfr\t$r, $s",
                  [(set AR:$r, (bitconvert FPR:$s))]>, Requires<[HasSingleFloat]> {
  let t = 0x04;
}

def ROUND_S : RRR_Inst<0x00, 0x0A, 0x08, (outs AR:$r), (ins FPR:$s, uimm4:$imm),
                      "round.s\t$r, $s, $imm", []>, Requires<[HasSingleFloat]> {
  bits<4> imm;

  let t = imm;
}

def RSQRT0_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
                       "rsqrt0.s\t$r, $s", []>, Requires<[HasSingleFloat]> {
  let t = 0x0A;
}

def SQRT0_S : RRR_Inst<0x00, 0x0A, 0x0F, (outs FPR:$r), (ins FPR:$s),
                      "sqrt0.s\t$r, $s", []>, Requires<[HasSingleFloat]> {
  let t = 0x09;
}

def TRUNC_S : RRR_Inst<0x00, 0x0A, 0x09, (outs AR:$r), (ins FPR:$s, uimm4:$imm),
                      "trunc.s\t$r, $s, $imm", []>, Requires<[HasSingleFloat]> {
  bits<4> imm;

  let t = imm;
}

def : Pat<(i32 (any_fp_to_sint FPR:$s)), (TRUNC_S FPR:$s, 0)>;

def : Pat<(i32 (int_xtensa_xt_trunc_s f32:$frs, timm:$imm)),
          (TRUNC_S FPR:$frs, uimm4:$imm)>,
          Requires<[HasSingleFloat]>;

def UFLOAT_S : RRR_Inst<0x00, 0x0A, 0x0D, (outs FPR:$r), (ins AR:$s, uimm4:$imm),
                       "ufloat.s\t$r, $s, $imm", []>, Requires<[HasSingleFloat]> {
  bits<4> imm;

  let t = imm;
}

def : Pat<(f32 (uint_to_fp AR:$s)), (UFLOAT_S AR:$s, 0)>;

def : Pat<(f32 (int_xtensa_xt_ufloat_s i32:$s, timm:$imm)),
          (UFLOAT_S AR:$s, uimm4:$imm)>,
          Requires<[HasSingleFloat]>;

def UTRUNC_S : RRR_Inst<0x00, 0x0A, 0x0e, (outs AR:$r), (ins FPR:$s, uimm4:$imm),
                       "utrunc.s\t$r, $s, $imm", []>, Requires<[HasSingleFloat]> {
  bits<4> imm;

  let t = imm;
}

def : Pat<(i32 (any_fp_to_uint FPR:$s)), (UTRUNC_S FPR:$s, 0)>;

def : Pat<(i32 (int_xtensa_xt_utrunc_s f32:$frs, timm:$imm)),
          (UTRUNC_S FPR:$frs, uimm4:$imm)>,
          Requires<[HasSingleFloat]>;

def WFR : RRR_Inst<0x00, 0x0A, 0x0f, (outs FPR:$r), (ins AR:$s),
                  "wfr\t$r, $s",
                  [(set FPR:$r, (bitconvert AR:$s))]>, Requires<[HasSingleFloat]> {
  let t = 0x05;
}

// FP select operations
let usesCustomInserter = 1, Predicates = [HasSingleFloat] in {
  def SELECT_CC_FP_INT : Pseudo<(outs AR:$dst), (ins FPR:$lhs, FPR:$rhs, AR:$t, AR:$f, i32imm:$cond),
                               "!select_cc_fp_int $dst, $lhs, $rhs, $t, $f, $cond",
                               [(set AR:$dst, (Xtensa_select_cc_fp FPR:$lhs, FPR:$rhs, AR:$t, AR:$f, imm:$cond))]>;
  def SELECT_CC_INT_FP : Pseudo<(outs FPR:$dst), (ins AR:$lhs, AR:$rhs, FPR:$t, FPR:$f, i32imm:$cond),
                               "!select_cc_int_fp $dst, $lhs, $rhs, $t, $f, $cond",
                               [(set FPR:$dst, (Xtensa_select_cc_fp AR:$lhs, AR:$rhs, FPR:$t, FPR:$f, imm:$cond))]>;
  def SELECT_CC_FP_FP : Pseudo<(outs FPR:$dst), (ins FPR:$lhs, FPR:$rhs, FPR:$t, FPR:$f, i32imm:$cond),
                              "!select_cc_fp_fp $dst, $lhs, $rhs, $t, $f, $cond",
                              [(set FPR:$dst, (Xtensa_select_cc_fp FPR:$lhs, FPR:$rhs, FPR:$t, FPR:$f, imm:$cond))]>;
}

// FP brcc pesudo operation
let usesCustomInserter = 1, isBranch = 1, isTerminator = 1, isBarrier = 1, Predicates = [HasSingleFloat] in {
  def BRCC_FP : Pseudo<(outs), (ins i32imm:$cond, FPR:$lhs, FPR:$rhs, brtarget:$target),
                      "!brcc_fp $cond, $lhs, $rhs, $target",
                      [(Xtensa_brcc_fp imm:$cond, FPR:$lhs, FPR:$rhs, bb:$target)]>;
}

//===----------------------------------------------------------------------===//
// Loop Instructions
//===----------------------------------------------------------------------===//

def LOOP : RRI8_Inst<0x06, (outs), (ins AR:$s, ltarget:$target),
                    "loop\t$s, $target", []>, Requires<[HasLoop]> {
  bits<8> target;

  let r = 0x08;
  let t = 0x07;
  let imm8 = target;
}

def LOOPGTZ : RRI8_Inst<0x06, (outs), (ins AR:$s, ltarget:$target),
                       "loopgtz\t$s, $target", []>, Requires<[HasLoop]> {
  bits<8> target;

  let r = 0x0A;
  let t = 0x07;
  let imm8 = target;
}

def LOOPNEZ : RRI8_Inst<0x06, (outs), (ins AR:$s, ltarget:$target),
                       "loopnez\t$s, $target", []>, Requires<[HasLoop]> {
  bits<8> target;

  let r = 0x09;
  let t = 0x07;
  let imm8 = target;
}

let isTerminator = 1, isBarrier = 1, hasSideEffects = 1, Size = 3 in {
  def LOOPINIT : Pseudo<(outs AR:$elts), (ins AR:$eltsin),
                       "!loopinit $elts, $eltsin", [(set AR:$elts, (int_start_loop_iterations AR:$eltsin))]>;
}

// LOOPSTART pseudo instruction reserves 9 bytes for LOOP operation and NOP operations for possible alignment.
let isTerminator = 1, isBarrier = 1, hasSideEffects = 1, Size = 9 in {
  def LOOPSTART : Pseudo<(outs), (ins AR:$s, brtarget:$target),
                        "!loopstart $s, $target", []>;
}

// LOOPEND pseudo instruction reserves 6 bytes for Jump and NOP operations.
let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 0, Size = 6 in {
  def LOOPEND : Pseudo<(outs), (ins brtarget:$target),
                      "!loopend $target", [(Xtensa_loopend bb:$target)]>;
}

let isTerminator = 1, isBarrier = 1, hasSideEffects = 1, Size = 3 in {
  def LOOPDEC : Pseudo<(outs AR:$eltsout), (ins AR:$eltsin),
                      "!loopdec $eltsout, $eltsin", [(set AR:$eltsout, (Xtensa_loopdec AR:$eltsin))]>;
}

let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 0, Size = 3 in {
  def LOOPBR : Pseudo<(outs), (ins AR:$elts, brtarget:$target),
                     "!loopbr $elts, $target", [(Xtensa_loopbr AR:$elts, bb:$target)]>;
}

//===----------------------------------------------------------------------===//
// SEXT Instructions
//===----------------------------------------------------------------------===//

def SEXT : RRR_Inst<0x00, 0x03, 0x02, (outs AR:$r), (ins AR:$s, seimm7_22:$imm),
                   "sext\t$r, $s, $imm", []>, Requires<[HasSEXT]> {
  bits<4> imm;

  let t = imm;
}

//===----------------------------------------------------------------------===//
// NSA Instructions
//===----------------------------------------------------------------------===//

def NSA : RRR_Inst<0x00, 0x00, 0x04, (outs AR:$t), (ins AR:$s),
                  "nsa\t$t, $s", []>, Requires<[HasNSA]> {
  let r = 0xE;
}

def NSAU : RRR_Inst<0x00, 0x00, 0x04, (outs AR:$t), (ins AR:$s),
                   "nsau\t$t, $s", []>, Requires<[HasNSA]> {
  let r = 0xF;
}

//===----------------------------------------------------------------------===//
// Mul16 Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasMul16] in {
  def MUL16S : RRR_Inst<0x00, 0x01, 0x0D, (outs AR:$r), (ins AR:$s, AR:$t),
                       "mul16s\t$r, $s, $t", []>;
  def MUL16U : RRR_Inst<0x00, 0x01, 0x0C, (outs AR:$r), (ins AR:$s, AR:$t),
                       "mul16u\t$r, $s, $t", []>;
}

//===----------------------------------------------------------------------===//
// Mul32 Instructions
//===----------------------------------------------------------------------===//

def MULL  : ArithLogic_RRR<0x08, 0x02, "mull", mul, 1>, Requires<[HasMul32]>;
def MULUH : ArithLogic_RRR<0x0A, 0x02, "muluh", mulhu, 1>, Requires<[HasMul32High]>;
def MULSH : ArithLogic_RRR<0x0B, 0x02, "mulsh", mulhs, 1>, Requires<[HasMul32High]>;

//===----------------------------------------------------------------------===//
// Div32 Instructions
//===----------------------------------------------------------------------===//

def QUOS : ArithLogic_RRR<0x0D, 0x02, "quos", sdiv>, Requires<[HasDiv32]>;
def QUOU : ArithLogic_RRR<0x0C, 0x02, "quou", udiv>, Requires<[HasDiv32]>;
def REMS : ArithLogic_RRR<0x0F, 0x02, "rems", srem>, Requires<[HasDiv32]>;
def REMU : ArithLogic_RRR<0x0E, 0x02, "remu", urem>, Requires<[HasDiv32]>;

//===----------------------------------------------------------------------===//
// S32C1I
//===----------------------------------------------------------------------===//

let mayStore = 1, mayLoad = 1, Predicates = [HasS32C1I] in {
  def S32C1I : RRI8_Inst<0x02, (outs AR:$a), (ins AR:$t, mem32:$addr),
                        "s32c1i\t$t, $addr", []> {
    bits<12> addr;

    let r = 0x0e;
    let Uses = [SCOMPARE1];
    let Constraints = "$a = $t";
    let imm8{7-0} = addr{11-4};
    let s{3-0} = addr{3-0};
  }
}

//===----------------------------------------------------------------------===//
// Debug instructions
//===----------------------------------------------------------------------===//

let isBarrier = 1, isTerminator = 1 in {
  def BREAK : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins uimm4:$s, uimm4:$t),
                      "break\t$s, $t", []>, Requires<[HasDebug]> {
    let r = 0x04;
  }

  def BREAK_N : RRRN_Inst<0x0C, (outs), (ins uimm4:$imm),
                         "break.n\t$imm", []>, Requires<[HasDensity, HasDebug]> {
    bits<4> imm;

    let r = 0xf;
    let s = imm;
    let t = 0x2;
  }
}

def : Pat<(trap), (BREAK (i32 1), (i32 15))>;

//===----------------------------------------------------------------------===//
// Exception feature instructions
//===----------------------------------------------------------------------===//

def EXCW : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                   "excw", []>, Requires<[HasException]> {
  let r = 0x2;
  let s = 0x0;
  let t = 0x8;
}

def RFDE : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                   "rfde", []>, Requires<[HasException]> {
  let r = 0x3;
  let s = 0x2;
  let t = 0x0;
}


def RFE : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                  "rfe", []>, Requires<[HasException]> {
  let r = 0x3;
  let s = 0x0;
  let t = 0x0;
}

def SYSCALL : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins),
                      "syscall", []>, Requires<[HasException]> {
  let r = 0x5;
  let s = 0x0;
  let t = 0x0;
}

//===----------------------------------------------------------------------===//
// Interrupt feature instructions
//===----------------------------------------------------------------------===//

def RSIL : RRR_Inst<0x00, 0x00, 0x00, (outs AR:$t), (ins uimm4:$imm),
                   "rsil\t$t, $imm", []>, Requires<[HasInterrupt]> {
  bits<4> imm;

  let r = 0x6;
  let s = imm{3-0};
}

def WAITI : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins uimm4:$imm),
                   "waiti\t$imm", []>, Requires<[HasInterrupt]> {
  bits<4> imm;

  let r = 0x7;
  let s = imm{3-0};
  let t = 0;
}

def RFI : RRR_Inst<0x00, 0x00, 0x00, (outs), (ins uimm4:$imm),
                  "rfi\t$imm", []>, Requires<[HasInterrupt]> {
  bits<4> imm;

  let r = 0x3;
  let s = imm{3-0};
  let t = 0x1;
}

//===----------------------------------------------------------------------===//
// Region Protection feature instructions
//===----------------------------------------------------------------------===//

def WDTLB : RRR_Inst<0x00, 0x00, 0x05, (outs AR:$t), (ins AR:$s),
                   "wdtlb\t$t, $s", []>, Requires<[HasRegionProtection]> {
  let r = 0xE;
}

def WITLB : RRR_Inst<0x00, 0x00, 0x05, (outs AR:$t), (ins AR:$s),
                    "witlb\t$t, $s", []>, Requires<[HasRegionProtection]> {
  let r = 0x6;
}

//===----------------------------------------------------------------------===//
// Illegal instructions
//===----------------------------------------------------------------------===//

let isBarrier = 1, isTerminator = 1 in {
  def ILL : CALLX_Inst<0x00, 0x00, 0x00, (outs), (ins),
                      "ill", []> {
    let m = 0x0;
    let n = 0x0;
    let r = 0;
    let s = 0;
  }

  def ILL_N : RRRN_Inst<0x0D, (outs), (ins),
                       "ill.n", []>, Requires<[HasDensity]> {
    let r = 0xf;
    let s = 0x0;
    let t = 0x6;
  }
}

//===----------------------------------------------------------------------===//
// Atomic patterns
//===----------------------------------------------------------------------===//

def : Pat<(i32 (atomic_load_8  addr_ish1:$addr)), (L8UI addr_ish1:$addr)>;
def : Pat<(i32 (atomic_load_16 addr_ish2:$addr)), (L16UI addr_ish2:$addr)>;
def : Pat<(i32 (atomic_load_32 addr_ish4:$addr)), (L32I addr_ish4:$addr)>;

def : Pat<(atomic_store_8  addr_ish1:$addr, AR:$t), (S8I AR:$t, addr_ish1:$addr)>;
def : Pat<(atomic_store_16 addr_ish2:$addr, AR:$t), (S16I AR:$t, addr_ish2:$addr)>;
def : Pat<(atomic_store_32 addr_ish4:$addr, AR:$t), (S32I AR:$t, addr_ish4:$addr)>;

let usesCustomInserter = 1, Predicates = [HasS32C1I] in {
  def ATOMIC_CMP_SWAP_8_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$cmp, AR:$swap),
                                  "!atomic_cmp_swap_8_p, $dst, $ptr, $cmp, $swap",
                                  [(set AR:$dst, (atomic_cmp_swap_8 AR:$ptr, AR:$cmp, AR:$swap))]>;
  def ATOMIC_CMP_SWAP_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$cmp, AR:$swap),
                                   "!atomic_cmp_swap_16_p, $dst, $ptr, $cmp, $swap",
                                   [(set AR:$dst, (atomic_cmp_swap_16 AR:$ptr, AR:$cmp, AR:$swap))]>;
  def ATOMIC_CMP_SWAP_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$cmp, AR:$swap),
                                   "!atomic_cmp_swap_32_p, $dst, $ptr, $cmp, $swap",
                                   [(set AR:$dst, (atomic_cmp_swap_32 AR:$ptr, AR:$cmp, AR:$swap))]>;

  def ATOMIC_SWAP_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$swap),
                               "!atomic_swap_8_p, $dst, $ptr, $swap",
                               [(set AR:$dst, (atomic_swap_8 AR:$ptr, AR:$swap))]>;
  def ATOMIC_SWAP_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$swap),
                               "!atomic_swap_16_p, $dst, $ptr, $swap",
                               [(set AR:$dst, (atomic_swap_16 AR:$ptr, AR:$swap))]>;
  def ATOMIC_SWAP_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$swap),
                               "!atomic_swap_32_p, $dst, $ptr, $swap",
                               [(set AR:$dst, (atomic_swap_32 AR:$ptr, AR:$swap))]>;

  def ATOMIC_LOAD_ADD_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_add_8_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_add_8 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_ADD_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_add_16_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_add_16 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_ADD_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_add_32_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_add_32 AR:$ptr, AR:$arg))]>;

  def ATOMIC_LOAD_SUB_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_sub_8_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_sub_8 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_SUB_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_sub_16_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_sub_16 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_SUB_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_sub_32_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_sub_32 AR:$ptr, AR:$arg))]>;

  def ATOMIC_LOAD_AND_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_and_8_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_and_8 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_AND_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_and_16_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_and_16 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_AND_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_and_32_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_and_32 AR:$ptr, AR:$arg))]>;

  def ATOMIC_LOAD_OR_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_or_8_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_or_8 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_OR_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_or_16_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_or_16 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_OR_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_or_32_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_or_32 AR:$ptr, AR:$arg))]>;

  def ATOMIC_LOAD_XOR_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_xor_8_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_xor_8 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_XOR_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_xor_16_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_xor_16 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_XOR_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_xor_32_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_xor_32 AR:$ptr, AR:$arg))]>;

  def ATOMIC_LOAD_NAND_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                    "!atomic_load_nand_8_p, $dst, $ptr, $arg",
                                    [(set AR:$dst, (atomic_load_nand_8 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_NAND_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                    "!atomic_load_nand_16_p, $dst, $ptr, $arg",
                                    [(set AR:$dst, (atomic_load_nand_16 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_NAND_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                    "!atomic_load_nand_32_p, $dst, $ptr, $arg",
                                    [(set AR:$dst, (atomic_load_nand_32 AR:$ptr, AR:$arg))]>;

  def ATOMIC_LOAD_MIN_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_min_8_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_min_8 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_MIN_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_min_16_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_min_16 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_MIN_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_min_32_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_min_32 AR:$ptr, AR:$arg))]>;			
										
  def ATOMIC_LOAD_MAX_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_max_8_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_max_8 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_MAX_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_max_16_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_max_16 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_MAX_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                   "!atomic_load_max_32_p, $dst, $ptr, $arg",
                                   [(set AR:$dst, (atomic_load_max_32 AR:$ptr, AR:$arg))]>;	

  def ATOMIC_LOAD_UMIN_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                    "!atomic_load_umin_8_p, $dst, $ptr, $arg",
                                    [(set AR:$dst, (atomic_load_umin_8 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_UMIN_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                    "!atomic_load_umin_16_p, $dst, $ptr, $arg",
                                    [(set AR:$dst, (atomic_load_umin_16 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_UMIN_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                    "!atomic_load_umin_32_p, $dst, $ptr, $arg",
                                    [(set AR:$dst, (atomic_load_umin_32 AR:$ptr, AR:$arg))]>;			
										
  def ATOMIC_LOAD_UMAX_8_P  : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                    "!atomic_load_umax_8_p, $dst, $ptr, $arg",
                                    [(set AR:$dst, (atomic_load_umax_8 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_UMAX_16_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                    "!atomic_load_umax_16_p, $dst, $ptr, $arg",
                                    [(set AR:$dst, (atomic_load_umax_16 AR:$ptr, AR:$arg))]>;
  def ATOMIC_LOAD_UMAX_32_P : Pseudo<(outs AR:$dst), (ins AR:$ptr, AR:$arg),
                                    "!atomic_load_umax_32_p, $dst, $ptr, $arg",
                                    [(set AR:$dst, (atomic_load_umax_32 AR:$ptr, AR:$arg))]>;	
}

//===----------------------------------------------------------------------===//
// Xtensa ESP32S2 Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasESP32S2Ops] in {
  def WR_MASK_GPIO_OUT : RRR_Inst<0x0, 0x06, 0x0, (outs), (ins AR:$s, AR:$t),
                                 "wr_mask_gpio_out\t$s, $t", []> {
    let r = 0x2;
  }

  def SET_BIT_GPIO_OUT : RRR_Inst<0x0, 0x06, 0x0, (outs), (ins select_256:$imm),
                                 "set_bit_gpio_out\t$imm", []> {
    bits<8> imm;

    let r = 0x1;
    let s = imm{7-4};
    let t = imm{3-0};
  }

  def CLR_BIT_GPIO_OUT : RRR_Inst<0x0, 0x06, 0x0, (outs), (ins select_256:$imm),
                                 "clr_bit_gpio_out\t$imm", []> {
    bits<8> imm;

    let r = 0x0;
    let s = imm{7-4};
    let t = imm{3-0};
  }

  def GET_GPIO_IN : RRR_Inst<0x0, 0x06, 0x0, (outs AR:$t), (ins),
                            "get_gpio_in\t$t", []> {
    let r = 0x3;
    let s = 0x0;
  }
}

//===----------------------------------------------------------------------===//
// Xtensa ESP32S3 Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasESP32S3Ops] in {
  def EE_WR_MASK_GPIO_OUT : RRR_Inst<0x04, 0x02, 0x07, (outs), (ins AR:$t, AR:$s),
                                    "ee.wr_mask_gpio_out\t$t, $s", []> {
    let r = 0x4;
  }

  def EE_SET_BIT_GPIO_OUT : RRR_Inst<0x04, 0x05, 0x07, (outs), (ins select_256:$imm),
                                "ee.set_bit_gpio_out\t$imm", []> {
    bits<8> imm;

    let r = 0x4;
    let s = imm{7-4};
    let t = imm{3-0};
  }

  def EE_CLR_BIT_GPIO_OUT : RRR_Inst<0x04, 0x06, 0x07, (outs), (ins select_256:$imm),
                                "ee.clr_bit_gpio_out\t$imm", []> {
    bits<8> imm;

    let r = 0x4;
    let s = imm{7-4};
    let t = imm{3-0};
  }

  def EE_GET_GPIO_IN : RRR_Inst<0x04, 0x05, 0x06, (outs AR:$t), (ins),
                               "ee.get_gpio_in\t$t", []> {
    let r = 0x0;
    let s = 0x8;
  }
}


//===----------------------------------------------------------------------===//
// FP intrinsic patterns
//===----------------------------------------------------------------------===//
let Predicates = [HasSingleFloat] in {

  def ADDEXP_S_PAT :Pat<(f32 (int_xtensa_xt_addexp_s FPR:$frr, FPR:$frs)),
  (ADDEXP_S FPR:$frr, FPR:$frs)>;

  def ADDEXPM_S_PAT :Pat<(f32 (int_xtensa_xt_addexpm_s FPR:$frr, FPR:$frs)),
  (ADDEXPM_S FPR:$frr, FPR:$frs)>;

  def CEIL_S_PAT :Pat<(i32 (int_xtensa_xt_ceil_s FPR:$frs, timm:$imm_t)),
  (CEIL_S FPR:$frs, timm:$imm_t)>;

  def DIV0_S_PAT :Pat<(f32 (int_xtensa_xt_div0_s FPR:$frs)),
  (DIV0_S FPR:$frs)>;

  def DIVN_S_PAT :Pat<(f32 (int_xtensa_xt_divn_s FPR:$frr, FPR:$frs, FPR:$frt)),
  (DIVN_S FPR:$frr, FPR:$frs, FPR:$frt)>;

  def FLOOR_S_PAT :Pat<(i32 (int_xtensa_xt_floor_s FPR:$frs, timm:$imm_t)),
  (FLOOR_S FPR:$frs, timm:$imm_t)>;

  def LSI_PAT :Pat<(f32 (int_xtensa_xt_lsi AR:$ars, timm:$imm8x4)),
  (LSI AR:$ars, timm:$imm8x4)>;

  def LSX_PAT :Pat<(f32 (int_xtensa_xt_lsx AR:$ars, AR:$art)),
  (LSX AR:$ars, AR:$art)>;

  def MADD_S_PAT :Pat<(f32 (int_xtensa_xt_madd_s FPR:$frr, FPR:$frs, FPR:$frt)),
  (MADD_S FPR:$frr, FPR:$frs, FPR:$frt)>;

  def MADDN_S_PAT :Pat<(f32 (int_xtensa_xt_maddn_s FPR:$frr, FPR:$frs, FPR:$frt)),
  (MADDN_S FPR:$frr, FPR:$frs, FPR:$frt)>;

  def MKDADJ_S_PAT :Pat<(f32 (int_xtensa_xt_mkdadj_s FPR:$frr, FPR:$frs)),
  (MKDADJ_S FPR:$frr, FPR:$frs)>;

  def MKSADJ_S_PAT :Pat<(f32 (int_xtensa_xt_mksadj_s FPR:$frs)),
  (MKSADJ_S FPR:$frs)>;

  def MOV_S_PAT :Pat<(f32 (int_xtensa_xt_mov_s FPR:$frs)),
  (MOV_S FPR:$frs)>;

  def MOVEQZ_S_PAT :Pat<(f32 (int_xtensa_xt_moveqz_s FPR:$frr, FPR:$frs, AR:$art)),
  (MOVEQZ_S FPR:$frr, FPR:$frs, AR:$art)>;

  def MOVGEZ_S_PAT :Pat<(f32 (int_xtensa_xt_movgez_s FPR:$frr, FPR:$frs, AR:$art)),
  (MOVGEZ_S FPR:$frr, FPR:$frs, AR:$art)>;

  def MOVLTZ_S_PAT :Pat<(f32 (int_xtensa_xt_movltz_s FPR:$frr, FPR:$frs, AR:$art)),
  (MOVLTZ_S FPR:$frr, FPR:$frs, AR:$art)>;

  def MOVNEZ_S_PAT :Pat<(f32 (int_xtensa_xt_movnez_s FPR:$frr, FPR:$frs, AR:$art)),
  (MOVNEZ_S FPR:$frr, FPR:$frs, AR:$art)>;

  def MSUB_S_PAT :Pat<(f32 (int_xtensa_xt_msub_s FPR:$frr, FPR:$frs, FPR:$frt)),
  (MSUB_S FPR:$frr, FPR:$frs, FPR:$frt)>;

  def NEG_S_PAT :Pat<(f32 (int_xtensa_xt_neg_s FPR:$frs)),
  (NEG_S FPR:$frs)>;

  def NEXP01_S_PAT :Pat<(f32 (int_xtensa_xt_nexp01_s FPR:$frs)),
  (NEXP01_S FPR:$frs)>;

  def RECIP0_S_PAT :Pat<(f32 (int_xtensa_xt_recip0_s FPR:$frs)),
  (RECIP0_S FPR:$frs)>;

  def RFR_PAT :Pat<(i32 (int_xtensa_xt_rfr FPR:$frs)),
  (RFR FPR:$frs)>;

  def ROUND_S_PAT :Pat<(i32 (int_xtensa_xt_round_s FPR:$frs, timm:$imm_t)),
  (ROUND_S FPR:$frs, timm:$imm_t)>;

  def RSQRT0_S_PAT :Pat<(f32 (int_xtensa_xt_rsqrt0_s FPR:$frs)),
  (RSQRT0_S FPR:$frs)>;

  def RUR_FCR_PAT :Pat<(i32 (int_xtensa_xt_rur_fcr )),
  (RUR FCR)>;

  def RUR_FSR_PAT :Pat<(i32 (int_xtensa_xt_rur_fsr )),
  (RUR FSR )>;

  def SQRT0_S_PAT :Pat<(f32 (int_xtensa_xt_sqrt0_s FPR:$frs)),
  (SQRT0_S FPR:$frs)>;

  def SSI_PAT :Pat<(int_xtensa_xt_ssi FPR:$frt, AR:$ars, timm:$imm8x4),
  (SSI FPR:$frt, AR:$ars, timm:$imm8x4)>;

  def SSIP_PAT :Pat<(i32 (int_xtensa_xt_ssip FPR:$frt, AR:$ars, timm:$imm8x4)),
  (SSIP FPR:$frt, AR:$ars, timm:$imm8x4)>;

  def SSX_PAT :Pat<(int_xtensa_xt_ssx FPR:$frr, AR:$ars, AR:$art),
  (SSX FPR:$frr, AR:$ars, AR:$art)>;

  def SSXP_PAT :Pat<(i32 (int_xtensa_xt_ssxp FPR:$frr, AR:$ars, AR:$art)),
  (SSXP FPR:$frr, AR:$ars, AR:$art)>;

  def WFR_PAT :Pat<(f32 (int_xtensa_xt_wfr AR:$ars)),
  (WFR AR:$ars)>;

  def WUR_FCR_PAT :Pat<(int_xtensa_xt_wur_fcr AR:$art),
  (WUR_FCR AR:$art)>;

  def WUR_FSR_PAT :Pat<(int_xtensa_xt_wur_fsr AR:$art),
  (WUR_FSR AR:$art)>;

}
//===----------------------------------------------------------------------===//
// DSP Instructions
//===----------------------------------------------------------------------===//
include "XtensaDSPInstrInfo.td"

//===----------------------------------------------------------------------===//
// HiFi3 Instructions
//===----------------------------------------------------------------------===//
include "XtensaHIFIInstrFormats.td"
include "XtensaHIFIInstrInfo.td"
include "XtensaHIFIInstrPatterns.td"
