//===-- RISCVInstrInfoXTHeadMatrix.td ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the XTHeadMatrix (RVM 0.6) vendor extension defined by
// T-Head of Alibaba. It provides decoupled matrix instructions for AI/ML
// workloads including matrix multiply, load/store, configuration, data
// movement, and element-wise arithmetic.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand definitions
//===----------------------------------------------------------------------===//

def thrvmuimm10 : RISCVUImmOp<10>;
def thrvmuimm3  : RISCVUImmOp<3>;

//===----------------------------------------------------------------------===//
// Instruction format classes
//===----------------------------------------------------------------------===//
// All 32-bit instructions use OPC_CUSTOM_1 (0b0101011)
// Bit layout: [31:28]func4 [27:26]uop [25:23]ctrl/size_sup
//             [22:20]ms2 [19:18]s_size [17:15]ms1 [14:12]func3
//             [11:10]d_size [9:7]md [6:0]opcode

// Base class for all XTHeadMatrix 32-bit instructions
class THRVMInst<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  let Inst{6-0} = OPC_CUSTOM_1.Value;
  let DecoderNamespace = "XTHeadMatrix";
}

// Configuration with 10-bit immediate: msettile*i
// Layout: [31:28]func4 [27:26]00 [25]0(imm) [24:15]uimm10 [14:12]000 [11:7]00000
class THRVMInstCfgImm<bits<4> func4,
                       dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<10> imm;

  let Inst{31-28} = func4;
  let Inst{27-26} = 0b00;
  let Inst{25} = 0;        // ctrl=0: immediate mode
  let Inst{24-15} = imm;
  let Inst{14-12} = 0b000;
  let Inst{11-7} = 0b00000; // nop field
}

// Configuration with GPR source: msettile*
// Layout: [31:28]func4 [27:26]00 [25]1(reg) [24:20]00000 [19:15]rs1 [14:12]000 [11:7]00000
class THRVMInstCfgReg<bits<4> func4,
                       dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<5> rs1;

  let Inst{31-28} = func4;
  let Inst{27-26} = 0b00;
  let Inst{25} = 1;        // ctrl=1: register mode
  let Inst{24-20} = 0b00000;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-7} = 0b00000; // nop field
}

// Configuration: mrelease (no operands)
class THRVMInstCfgRelease<string opcodestr>
    : THRVMInst<(outs), (ins), opcodestr, ""> {
  let Inst{31-28} = 0b0000;
  let Inst{27-26} = 0b00;
  let Inst{25-7} = 0;
  let Inst{14-12} = 0b000;
}

// Load: md, (rs1), rs2
// Layout: [31:28]func4 [27:26]01 [25]0(load) [24:20]rs2 [19:15]rs1
//         [14:12]000 [11:10]d_size [9:7]md
class THRVMInstLoad<bits<4> func4, bits<2> dsize,
                     dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<5> rs1;
  bits<5> rs2;

  let Inst{31-28} = func4;
  let Inst{27-26} = 0b01;
  let Inst{25} = 0;        // ls=0: load
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = dsize;
  let Inst{9-7} = md;
  let mayLoad = 1;
  let mayStore = 0;
}

// Load whole-register: md, (rs1) — no rs2 (bits[24:20]=00000)
class THRVMInstLoadWhole<bits<4> func4, bits<2> dsize,
                          dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<5> rs1;

  let Inst{31-28} = func4;
  let Inst{27-26} = 0b01;
  let Inst{25} = 0;        // ls=0: load
  let Inst{24-20} = 0b00000; // no rs2
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = dsize;
  let Inst{9-7} = md;
  let mayLoad = 1;
  let mayStore = 0;
}

// Store: ms3, (rs1), rs2
// Layout: [31:28]func4 [27:26]01 [25]1(store) [24:20]rs2 [19:15]rs1
//         [14:12]000 [11:10]d_size [9:7]ms3
class THRVMInstStore<bits<4> func4, bits<2> dsize,
                      dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> ms3;
  bits<5> rs1;
  bits<5> rs2;

  let Inst{31-28} = func4;
  let Inst{27-26} = 0b01;
  let Inst{25} = 1;        // ls=1: store
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = dsize;
  let Inst{9-7} = ms3;
  let mayLoad = 0;
  let mayStore = 1;
}

// Store whole-register: ms3, (rs1) — no rs2 (bits[24:20]=00000)
class THRVMInstStoreWhole<bits<4> func4, bits<2> dsize,
                           dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> ms3;
  bits<5> rs1;

  let Inst{31-28} = func4;
  let Inst{27-26} = 0b01;
  let Inst{25} = 1;        // ls=1: store
  let Inst{24-20} = 0b00000; // no rs2
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = dsize;
  let Inst{9-7} = ms3;
  let mayLoad = 0;
  let mayStore = 1;
}

// Matrix multiply: md, ms2, ms1 (uop=10, func3=000)
class THRVMInstMMul<bits<4> func4, bits<3> size_sup, bits<2> ssize,
                     bits<2> dsize, dag outs, dag ins,
                     string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<3> ms2;
  bits<3> ms1;

  let Inst{31-28} = func4;
  let Inst{27-26} = 0b10;
  let Inst{25-23} = size_sup;
  let Inst{22-20} = ms2;
  let Inst{19-18} = ssize;
  let Inst{17-15} = ms1;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = dsize;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

// mzero: func4=0000, uop=11, imm3 in [25:23], all other fields zero except md
class THRVMInstMZero<bits<3> imm3, dag outs, dag ins,
                      string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;

  let Inst{31-28} = 0b0000;
  let Inst{27-26} = 0b11;
  let Inst{25-23} = imm3;
  let Inst{22-20} = 0b000;
  let Inst{19-18} = 0b00;
  let Inst{17-15} = 0b000;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = 0b00;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

// mmov.mm: func4=0001, uop=11
class THRVMInstMovMM<dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<3> ms1;

  let Inst{31-28} = 0b0001;
  let Inst{27-26} = 0b11;
  let Inst{25-23} = 0b000;
  let Inst{22-20} = 0b000;
  let Inst{19-18} = 0b00;
  let Inst{17-15} = ms1;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = 0b00;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

// mmov.x.m: func4=0010, uop=11 — matrix-to-GPR (type 2)
// [31:28]0010 [27:26]11 [25]0 [24:23]e_size [22:20]ms2 [19:15]rs1 [14:12]000 [11:7]rd
class THRVMInstMovXM<bits<2> esize, dag outs, dag ins,
                      string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<5> rd;
  bits<3> ms2;
  bits<5> rs1;

  let Inst{31-28} = 0b0010;
  let Inst{27-26} = 0b11;
  let Inst{25} = 0;
  let Inst{24-23} = esize;
  let Inst{22-20} = ms2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-7} = rd;
  let mayLoad = 0;
  let mayStore = 0;
}

// mmov.m.x: func4=0011, uop=11, bit[25]=1 — GPR-to-matrix (type 3)
// [31:28]0011 [27:26]11 [25]1 [24:20]rs2 [19:15]rs1 [14:12]000 [11:10]d_size [9:7]md
class THRVMInstMovMX<bits<2> dsize, dag outs, dag ins,
                      string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<5> rs2;
  bits<5> rs1;

  let Inst{31-28} = 0b0011;
  let Inst{27-26} = 0b11;
  let Inst{25} = 1;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = dsize;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

// mdup.m.x: func4=0011, uop=11, bit[25]=0 — GPR broadcast to matrix
// [31:28]0011 [27:26]11 [25]0 [24:20]rs2 [19:15]00000 [14:12]000 [11:10]d_size [9:7]md
class THRVMInstDupMX<bits<2> dsize, dag outs, dag ins,
                      string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<5> rs2;

  let Inst{31-28} = 0b0011;
  let Inst{27-26} = 0b11;
  let Inst{25} = 0;
  let Inst{24-20} = rs2;
  let Inst{19-15} = 0b00000;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = dsize;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

// mpack: func4=0100, uop=11
// [31:28]0100 [27:26]11 [25]0 [24:23]pack_mode [22:20]ms2 [19:18]00 [17:15]ms1 [14:12]000 [11:10]00 [9:7]md
class THRVMInstPack<bits<2> packmode, dag outs, dag ins,
                     string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<3> ms2;
  bits<3> ms1;

  let Inst{31-28} = 0b0100;
  let Inst{27-26} = 0b11;
  let Inst{25} = 0;
  let Inst{24-23} = packmode;
  let Inst{22-20} = ms2;
  let Inst{19-18} = 0b00;
  let Inst{17-15} = ms1;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = 0b00;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

// Slide: uimm3 in bits[25:23], ms2=000 unused
class THRVMInstSlide<bits<4> func4, bits<2> ssize, bits<2> dsize,
                      dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<3> ms1;
  bits<3> imm3;

  let Inst{31-28} = func4;
  let Inst{27-26} = 0b11;
  let Inst{25-23} = imm3;
  let Inst{22-20} = 0b000;
  let Inst{19-18} = ssize;
  let Inst{17-15} = ms1;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = dsize;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

// Broadcast: uimm3 in bits[25:23], ms2=000 unused
class THRVMInstBcast<bits<4> func4, bits<2> ssize, bits<2> dsize,
                      dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<3> ms1;
  bits<3> imm3;

  let Inst{31-28} = func4;
  let Inst{27-26} = 0b11;
  let Inst{25-23} = imm3;
  let Inst{22-20} = 0b000;
  let Inst{19-18} = ssize;
  let Inst{17-15} = ms1;
  let Inst{14-12} = 0b000;
  let Inst{11-10} = dsize;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

// Element-wise unary conversion: md, ms1 (func3=001)
class THRVMInstEWConv<bits<4> func4, bits<2> uop, bits<3> ctrl,
                       bits<2> ssize, bits<2> dsize,
                       dag outs, dag ins,
                       string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<3> ms1;

  let Inst{31-28} = func4;
  let Inst{27-26} = uop;
  let Inst{25-23} = ctrl;
  let Inst{22-20} = 0b000;
  let Inst{19-18} = ssize;
  let Inst{17-15} = ms1;
  let Inst{14-12} = 0b001;
  let Inst{11-10} = dsize;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

// Element-wise binary .mm: md, ms2, ms1 (func3=001, ctrl=111)
class THRVMInstEWBinMM<bits<4> func4, bits<2> uop,
                         bits<2> ssize, bits<2> dsize,
                         dag outs, dag ins,
                         string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<3> ms2;
  bits<3> ms1;

  let Inst{31-28} = func4;
  let Inst{27-26} = uop;
  let Inst{25-23} = 0b111;  // ctrl=111 for .mm
  let Inst{22-20} = ms2;
  let Inst{19-18} = ssize;
  let Inst{17-15} = ms1;
  let Inst{14-12} = 0b001;
  let Inst{11-10} = dsize;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

// Element-wise binary .mv.i: md, ms2, ms1, imm3 (func3=001)
// uimm3 in bits[25:23] (ctrl position)
class THRVMInstEWBinMVI<bits<4> func4, bits<2> uop,
                          bits<2> ssize, bits<2> dsize,
                          dag outs, dag ins,
                          string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<3> md;
  bits<3> ms2;
  bits<3> ms1;
  bits<3> imm3;

  let Inst{31-28} = func4;
  let Inst{27-26} = uop;
  let Inst{25-23} = imm3;   // uimm3 row index
  let Inst{22-20} = ms2;
  let Inst{19-18} = ssize;
  let Inst{17-15} = ms1;
  let Inst{14-12} = 0b001;
  let Inst{11-10} = dsize;
  let Inst{9-7} = md;
  let mayLoad = 0;
  let mayStore = 0;
}

//===----------------------------------------------------------------------===//
// Instruction definitions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXTHeadMatrix] in {

// All XTHeadMatrix instructions operate on implicit matrix register state,
// so they must be treated as having side effects to prevent the compiler
// from reordering, CSE-ing, or eliminating them.
let hasSideEffects = 1 in {

//===----------------------------------------------------------------------===//
// Configuration instructions (uop=00, func3=000)
//===----------------------------------------------------------------------===//

let mayLoad = 0, mayStore = 0 in {

def TH_MRELEASE : THRVMInstCfgRelease<"th.mrelease">;

def TH_MSETTILEKI : THRVMInstCfgImm<0b0001,
    (outs), (ins thrvmuimm10:$imm),
    "th.msettileki", "$imm">;

def TH_MSETTILEK : THRVMInstCfgReg<0b0001,
    (outs), (ins GPR:$rs1),
    "th.msettilek", "$rs1">;

def TH_MSETTILEMI : THRVMInstCfgImm<0b0010,
    (outs), (ins thrvmuimm10:$imm),
    "th.msettilemi", "$imm">;

def TH_MSETTILEM : THRVMInstCfgReg<0b0010,
    (outs), (ins GPR:$rs1),
    "th.msettilem", "$rs1">;

def TH_MSETTILENI : THRVMInstCfgImm<0b0011,
    (outs), (ins thrvmuimm10:$imm),
    "th.msettileni", "$imm">;

def TH_MSETTILEN : THRVMInstCfgReg<0b0011,
    (outs), (ins GPR:$rs1),
    "th.msettilen", "$rs1">;

} // mayLoad = 0, mayStore = 0

//===----------------------------------------------------------------------===//
// Load instructions (uop=01, ls=0, func3=000)
//===----------------------------------------------------------------------===//

multiclass THRVMLoadInst<string mnemonic, bits<4> func4> {
  def _E8  : THRVMInstLoad<func4, 0b00,
      (outs), (ins THRVMMR:$md, GPR:$rs1, GPR:$rs2),
      "th."#mnemonic#"e8", "$md, (${rs1}), $rs2">;
  def _E16 : THRVMInstLoad<func4, 0b01,
      (outs), (ins THRVMMR:$md, GPR:$rs1, GPR:$rs2),
      "th."#mnemonic#"e16", "$md, (${rs1}), $rs2">;
  def _E32 : THRVMInstLoad<func4, 0b10,
      (outs), (ins THRVMMR:$md, GPR:$rs1, GPR:$rs2),
      "th."#mnemonic#"e32", "$md, (${rs1}), $rs2">;
  def _E64 : THRVMInstLoad<func4, 0b11,
      (outs), (ins THRVMMR:$md, GPR:$rs1, GPR:$rs2),
      "th."#mnemonic#"e64", "$md, (${rs1}), $rs2">;
}

multiclass THRVMLoadWholeInst<string mnemonic, bits<4> func4> {
  def _E8  : THRVMInstLoadWhole<func4, 0b00,
      (outs), (ins THRVMMR:$md, GPR:$rs1),
      "th."#mnemonic#"e8", "$md, (${rs1})">;
  def _E16 : THRVMInstLoadWhole<func4, 0b01,
      (outs), (ins THRVMMR:$md, GPR:$rs1),
      "th."#mnemonic#"e16", "$md, (${rs1})">;
  def _E32 : THRVMInstLoadWhole<func4, 0b10,
      (outs), (ins THRVMMR:$md, GPR:$rs1),
      "th."#mnemonic#"e32", "$md, (${rs1})">;
  def _E64 : THRVMInstLoadWhole<func4, 0b11,
      (outs), (ins THRVMMR:$md, GPR:$rs1),
      "th."#mnemonic#"e64", "$md, (${rs1})">;
}

defm TH_MLAE  : THRVMLoadInst<"mla",  0b0000>;
defm TH_MLBE  : THRVMLoadInst<"mlb",  0b0001>;
defm TH_MLCE  : THRVMLoadInst<"mlc",  0b0010>;
defm TH_MLME  : THRVMLoadWholeInst<"mlm",  0b0011>;
defm TH_MLATE : THRVMLoadInst<"mlat", 0b0100>;
defm TH_MLBTE : THRVMLoadInst<"mlbt", 0b0101>;
defm TH_MLCTE : THRVMLoadInst<"mlct", 0b0110>;

//===----------------------------------------------------------------------===//
// Store instructions (uop=01, ls=1, func3=000)
//===----------------------------------------------------------------------===//

multiclass THRVMStoreInst<string mnemonic, bits<4> func4> {
  def _E8  : THRVMInstStore<func4, 0b00,
      (outs), (ins THRVMMR:$ms3, GPR:$rs1, GPR:$rs2),
      "th."#mnemonic#"e8", "$ms3, (${rs1}), $rs2">;
  def _E16 : THRVMInstStore<func4, 0b01,
      (outs), (ins THRVMMR:$ms3, GPR:$rs1, GPR:$rs2),
      "th."#mnemonic#"e16", "$ms3, (${rs1}), $rs2">;
  def _E32 : THRVMInstStore<func4, 0b10,
      (outs), (ins THRVMMR:$ms3, GPR:$rs1, GPR:$rs2),
      "th."#mnemonic#"e32", "$ms3, (${rs1}), $rs2">;
  def _E64 : THRVMInstStore<func4, 0b11,
      (outs), (ins THRVMMR:$ms3, GPR:$rs1, GPR:$rs2),
      "th."#mnemonic#"e64", "$ms3, (${rs1}), $rs2">;
}

multiclass THRVMStoreWholeInst<string mnemonic, bits<4> func4> {
  def _E8  : THRVMInstStoreWhole<func4, 0b00,
      (outs), (ins THRVMMR:$ms3, GPR:$rs1),
      "th."#mnemonic#"e8", "$ms3, (${rs1})">;
  def _E16 : THRVMInstStoreWhole<func4, 0b01,
      (outs), (ins THRVMMR:$ms3, GPR:$rs1),
      "th."#mnemonic#"e16", "$ms3, (${rs1})">;
  def _E32 : THRVMInstStoreWhole<func4, 0b10,
      (outs), (ins THRVMMR:$ms3, GPR:$rs1),
      "th."#mnemonic#"e32", "$ms3, (${rs1})">;
  def _E64 : THRVMInstStoreWhole<func4, 0b11,
      (outs), (ins THRVMMR:$ms3, GPR:$rs1),
      "th."#mnemonic#"e64", "$ms3, (${rs1})">;
}

defm TH_MSAE  : THRVMStoreInst<"msa",  0b0000>;
defm TH_MSBE  : THRVMStoreInst<"msb",  0b0001>;
defm TH_MSCE  : THRVMStoreInst<"msc",  0b0010>;
defm TH_MSME  : THRVMStoreWholeInst<"msm",  0b0011>;
defm TH_MSATE : THRVMStoreInst<"msat", 0b0100>;
defm TH_MSBTE : THRVMStoreInst<"msbt", 0b0101>;
defm TH_MSCTE : THRVMStoreInst<"msct", 0b0110>;

//===----------------------------------------------------------------------===//
// Matrix multiply instructions (uop=10, func3=000)
//===----------------------------------------------------------------------===//

// FP matmul (func4=0000)
def TH_MFMACC_H_E5    : THRVMInstMMul<0b0000, 0b000, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.h.e5", "$md, $ms2, $ms1">;
def TH_MFMACC_H_E4    : THRVMInstMMul<0b0000, 0b001, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.h.e4", "$md, $ms2, $ms1">;
def TH_MFMACC_BF16_E5 : THRVMInstMMul<0b0000, 0b100, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.bf16.e5", "$md, $ms2, $ms1">;
def TH_MFMACC_BF16_E4 : THRVMInstMMul<0b0000, 0b101, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.bf16.e4", "$md, $ms2, $ms1">;
def TH_MFMACC_S_E5    : THRVMInstMMul<0b0000, 0b000, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.s.e5", "$md, $ms2, $ms1">;
def TH_MFMACC_S_E4    : THRVMInstMMul<0b0000, 0b001, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.s.e4", "$md, $ms2, $ms1">;
def TH_MFMACC_H       : THRVMInstMMul<0b0000, 0b000, 0b01, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.h", "$md, $ms2, $ms1">;
def TH_MFMACC_S_H     : THRVMInstMMul<0b0000, 0b000, 0b01, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.s.h", "$md, $ms2, $ms1">;
def TH_MFMACC_S_BF16  : THRVMInstMMul<0b0000, 0b001, 0b01, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.s.bf16", "$md, $ms2, $ms1">;
def TH_MFMACC_S_TF32  : THRVMInstMMul<0b0000, 0b001, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.s.tf32", "$md, $ms2, $ms1">;
def TH_MFMACC_S       : THRVMInstMMul<0b0000, 0b000, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.s", "$md, $ms2, $ms1">;
def TH_MFMACC_D_S     : THRVMInstMMul<0b0000, 0b000, 0b10, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.d.s", "$md, $ms2, $ms1">;
def TH_MFMACC_D       : THRVMInstMMul<0b0000, 0b000, 0b11, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mfmacc.d", "$md, $ms2, $ms1">;

// Integer matmul (func4=0001)
def TH_MMACCU_W_B     : THRVMInstMMul<0b0001, 0b000, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mmaccu.w.b", "$md, $ms2, $ms1">;
def TH_MMACCUS_W_B    : THRVMInstMMul<0b0001, 0b001, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mmaccus.w.b", "$md, $ms2, $ms1">;
def TH_MMACCSU_W_B    : THRVMInstMMul<0b0001, 0b010, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mmaccsu.w.b", "$md, $ms2, $ms1">;
def TH_MMACC_W_B      : THRVMInstMMul<0b0001, 0b011, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mmacc.w.b", "$md, $ms2, $ms1">;
def TH_PMMACCU_W_B    : THRVMInstMMul<0b0001, 0b100, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.pmmaccu.w.b", "$md, $ms2, $ms1">;
def TH_PMMACCUS_W_B   : THRVMInstMMul<0b0001, 0b101, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.pmmaccus.w.b", "$md, $ms2, $ms1">;
def TH_PMMACCSU_W_B   : THRVMInstMMul<0b0001, 0b110, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.pmmaccsu.w.b", "$md, $ms2, $ms1">;
def TH_PMMACC_W_B     : THRVMInstMMul<0b0001, 0b111, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.pmmacc.w.b", "$md, $ms2, $ms1">;
def TH_MMACCU_D_H     : THRVMInstMMul<0b0001, 0b000, 0b01, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mmaccu.d.h", "$md, $ms2, $ms1">;
def TH_MMACCUS_D_H    : THRVMInstMMul<0b0001, 0b001, 0b01, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mmaccus.d.h", "$md, $ms2, $ms1">;
def TH_MMACCSU_D_H    : THRVMInstMMul<0b0001, 0b010, 0b01, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mmaccsu.d.h", "$md, $ms2, $ms1">;
def TH_MMACC_D_H      : THRVMInstMMul<0b0001, 0b011, 0b01, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mmacc.d.h", "$md, $ms2, $ms1">;

// Bypass matmul (func4=0010)
def TH_MMACCU_W_BP    : THRVMInstMMul<0b0010, 0b000, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mmaccu.w.bp", "$md, $ms2, $ms1">;
def TH_MMACC_W_BP     : THRVMInstMMul<0b0010, 0b011, 0b00, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mmacc.w.bp", "$md, $ms2, $ms1">;

//===----------------------------------------------------------------------===//
// Misc instructions (uop=11, func3=000)
//===----------------------------------------------------------------------===//

// mzero (func4=0000, distinguished by imm3 in bits[25:23])
def TH_MZERO   : THRVMInstMZero<0b000,
    (outs), (ins THRVMMR:$md),"th.mzero", "$md">;
def TH_MZERO2R : THRVMInstMZero<0b001,
    (outs), (ins THRVMMR:$md),"th.mzero2r", "$md">;
def TH_MZERO4R : THRVMInstMZero<0b011,
    (outs), (ins THRVMMR:$md),"th.mzero4r", "$md">;
def TH_MZERO8R : THRVMInstMZero<0b111,
    (outs), (ins THRVMMR:$md),"th.mzero8r", "$md">;

// mmov.mm (func4=0001)
def TH_MMOV_MM : THRVMInstMovMM<
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mmov.mm", "$md, $ms1">;

// mmov.x.m: matrix-to-GPR (func4=0010), 3 operands: rd, ms2, rs1
def TH_MMOVB_X_M : THRVMInstMovXM<0b00,
    (outs GPR:$rd), (ins THRVMMR:$ms2, GPR:$rs1),
    "th.mmovb.x.m", "$rd, $ms2, $rs1">;
def TH_MMOVH_X_M : THRVMInstMovXM<0b01,
    (outs GPR:$rd), (ins THRVMMR:$ms2, GPR:$rs1),
    "th.mmovh.x.m", "$rd, $ms2, $rs1">;
def TH_MMOVW_X_M : THRVMInstMovXM<0b10,
    (outs GPR:$rd), (ins THRVMMR:$ms2, GPR:$rs1),
    "th.mmovw.x.m", "$rd, $ms2, $rs1">;
def TH_MMOVD_X_M : THRVMInstMovXM<0b11,
    (outs GPR:$rd), (ins THRVMMR:$ms2, GPR:$rs1),
    "th.mmovd.x.m", "$rd, $ms2, $rs1">;

// mmov.m.x: GPR-to-matrix (func4=0011, bit[25]=1), 3 operands: md, rs2, rs1
def TH_MMOVB_M_X : THRVMInstMovMX<0b00,
    (outs), (ins THRVMMR:$md, GPR:$rs2, GPR:$rs1),
    "th.mmovb.m.x", "$md, $rs2, $rs1">;
def TH_MMOVH_M_X : THRVMInstMovMX<0b01,
    (outs), (ins THRVMMR:$md, GPR:$rs2, GPR:$rs1),
    "th.mmovh.m.x", "$md, $rs2, $rs1">;
def TH_MMOVW_M_X : THRVMInstMovMX<0b10,
    (outs), (ins THRVMMR:$md, GPR:$rs2, GPR:$rs1),
    "th.mmovw.m.x", "$md, $rs2, $rs1">;
def TH_MMOVD_M_X : THRVMInstMovMX<0b11,
    (outs), (ins THRVMMR:$md, GPR:$rs2, GPR:$rs1),
    "th.mmovd.m.x", "$md, $rs2, $rs1">;

// mdup.m.x: GPR broadcast to matrix (func4=0011, bit[25]=0), 2 operands: md, rs2
def TH_MDUPB_M_X : THRVMInstDupMX<0b00,
    (outs), (ins THRVMMR:$md, GPR:$rs2),
    "th.mdupb.m.x", "$md, $rs2">;
def TH_MDUPH_M_X : THRVMInstDupMX<0b01,
    (outs), (ins THRVMMR:$md, GPR:$rs2),
    "th.mduph.m.x", "$md, $rs2">;
def TH_MDUPW_M_X : THRVMInstDupMX<0b10,
    (outs), (ins THRVMMR:$md, GPR:$rs2),
    "th.mdupw.m.x", "$md, $rs2">;
def TH_MDUPD_M_X : THRVMInstDupMX<0b11,
    (outs), (ins THRVMMR:$md, GPR:$rs2),
    "th.mdupd.m.x", "$md, $rs2">;

// Pack (func4=0100)
def TH_MPACK   : THRVMInstPack<0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mpack", "$md, $ms2, $ms1">;
def TH_MPACKHL : THRVMInstPack<0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mpackhl", "$md, $ms2, $ms1">;
def TH_MPACKHH : THRVMInstPack<0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mpackhh", "$md, $ms2, $ms1">;

// Row slides (func4=0101/0110)
def TH_MRSLIDEDOWN : THRVMInstSlide<0b0101, 0b00, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mrslidedown", "$md, $ms1, $imm3">;
def TH_MRSLIDEUP : THRVMInstSlide<0b0110, 0b00, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mrslideup", "$md, $ms1, $imm3">;

// Column slides (func4=0111/1000)
def TH_MCSLIDEDOWN_B : THRVMInstSlide<0b0111, 0b00, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcslidedown.b", "$md, $ms1, $imm3">;
def TH_MCSLIDEDOWN_H : THRVMInstSlide<0b0111, 0b01, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcslidedown.h", "$md, $ms1, $imm3">;
def TH_MCSLIDEDOWN_W : THRVMInstSlide<0b0111, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcslidedown.w", "$md, $ms1, $imm3">;
def TH_MCSLIDEDOWN_D : THRVMInstSlide<0b0111, 0b11, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcslidedown.d", "$md, $ms1, $imm3">;

def TH_MCSLIDEUP_B : THRVMInstSlide<0b1000, 0b00, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcslideup.b", "$md, $ms1, $imm3">;
def TH_MCSLIDEUP_H : THRVMInstSlide<0b1000, 0b01, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcslideup.h", "$md, $ms1, $imm3">;
def TH_MCSLIDEUP_W : THRVMInstSlide<0b1000, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcslideup.w", "$md, $ms1, $imm3">;
def TH_MCSLIDEUP_D : THRVMInstSlide<0b1000, 0b11, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcslideup.d", "$md, $ms1, $imm3">;

// Broadcast (func4=1001/1010)
def TH_MRBCA_MV_I : THRVMInstBcast<0b1001, 0b00, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mrbca.mv.i", "$md, $ms1, $imm3">;

def TH_MCBCAB_MV_I : THRVMInstBcast<0b1010, 0b00, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcbcab.mv.i", "$md, $ms1, $imm3">;
def TH_MCBCAH_MV_I : THRVMInstBcast<0b1010, 0b01, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcbcah.mv.i", "$md, $ms1, $imm3">;
def TH_MCBCAW_MV_I : THRVMInstBcast<0b1010, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcbcaw.mv.i", "$md, $ms1, $imm3">;
def TH_MCBCAD_MV_I : THRVMInstBcast<0b1010, 0b11, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mcbcad.mv.i", "$md, $ms1, $imm3">;

//===----------------------------------------------------------------------===//
// Element-wise instructions (func3=001)
//===----------------------------------------------------------------------===//

//--- FP format conversions (uop=00, func4=0000) ---

// FP8 widening to FP16
def TH_MFCVTL_H_E4 : THRVMInstEWConv<0b0000, 0b00, 0b000, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.h.e4", "$md, $ms1">;
def TH_MFCVTH_H_E4 : THRVMInstEWConv<0b0000, 0b00, 0b010, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.h.e4", "$md, $ms1">;
def TH_MFCVTL_H_E5 : THRVMInstEWConv<0b0000, 0b00, 0b001, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.h.e5", "$md, $ms1">;
def TH_MFCVTH_H_E5 : THRVMInstEWConv<0b0000, 0b00, 0b011, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.h.e5", "$md, $ms1">;

// FP16 narrowing to FP8 (with l/h variants)
def TH_MFCVTL_E4_H : THRVMInstEWConv<0b0000, 0b00, 0b000, 0b01, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.e4.h", "$md, $ms1">;
def TH_MFCVTH_E4_H : THRVMInstEWConv<0b0000, 0b00, 0b010, 0b01, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.e4.h", "$md, $ms1">;
def TH_MFCVTL_E5_H : THRVMInstEWConv<0b0000, 0b00, 0b001, 0b01, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.e5.h", "$md, $ms1">;
def TH_MFCVTH_E5_H : THRVMInstEWConv<0b0000, 0b00, 0b011, 0b01, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.e5.h", "$md, $ms1">;

// FP16/BF16 widening to FP32
def TH_MFCVTL_S_H : THRVMInstEWConv<0b0000, 0b00, 0b000, 0b01, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.s.h", "$md, $ms1">;
def TH_MFCVTH_S_H : THRVMInstEWConv<0b0000, 0b00, 0b010, 0b01, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.s.h", "$md, $ms1">;
def TH_MFCVTL_S_BF16 : THRVMInstEWConv<0b0000, 0b00, 0b001, 0b01, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.s.bf16", "$md, $ms1">;
def TH_MFCVTH_S_BF16 : THRVMInstEWConv<0b0000, 0b00, 0b011, 0b01, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.s.bf16", "$md, $ms1">;

// FP32 narrowing to FP8 (with l/h variants)
def TH_MFCVTL_E4_S : THRVMInstEWConv<0b0000, 0b00, 0b000, 0b10, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.e4.s", "$md, $ms1">;
def TH_MFCVTH_E4_S : THRVMInstEWConv<0b0000, 0b00, 0b010, 0b10, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.e4.s", "$md, $ms1">;
def TH_MFCVTL_E5_S : THRVMInstEWConv<0b0000, 0b00, 0b100, 0b10, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.e5.s", "$md, $ms1">;
def TH_MFCVTH_E5_S : THRVMInstEWConv<0b0000, 0b00, 0b110, 0b10, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.e5.s", "$md, $ms1">;

// FP32 narrowing to FP16/BF16 (with l/h variants)
def TH_MFCVTL_H_S : THRVMInstEWConv<0b0000, 0b00, 0b000, 0b10, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.h.s", "$md, $ms1">;
def TH_MFCVTH_H_S : THRVMInstEWConv<0b0000, 0b00, 0b010, 0b10, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.h.s", "$md, $ms1">;
def TH_MFCVTL_BF16_S : THRVMInstEWConv<0b0000, 0b00, 0b100, 0b10, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.bf16.s", "$md, $ms1">;
def TH_MFCVTH_BF16_S : THRVMInstEWConv<0b0000, 0b00, 0b110, 0b10, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.bf16.s", "$md, $ms1">;

// TF32 conversions
def TH_MFCVT_TF32_S : THRVMInstEWConv<0b0000, 0b00, 0b110, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvt.tf32.s", "$md, $ms1">;
def TH_MFCVT_S_TF32 : THRVMInstEWConv<0b0000, 0b00, 0b001, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvt.s.tf32", "$md, $ms1">;

// FP32/FP64 widening/narrowing
def TH_MFCVTL_D_S : THRVMInstEWConv<0b0000, 0b00, 0b000, 0b10, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.d.s", "$md, $ms1">;
def TH_MFCVTH_D_S : THRVMInstEWConv<0b0000, 0b00, 0b010, 0b10, 0b11,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.d.s", "$md, $ms1">;
def TH_MFCVTL_S_D : THRVMInstEWConv<0b0000, 0b00, 0b000, 0b11, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvtl.s.d", "$md, $ms1">;
def TH_MFCVTH_S_D : THRVMInstEWConv<0b0000, 0b00, 0b010, 0b11, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfcvth.s.d", "$md, $ms1">;

//--- Float-int conversions (uop=00, func4=0001) ---

// INT8 to FP16
def TH_MSFCVTL_H_B : THRVMInstEWConv<0b0001, 0b00, 0b001, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.msfcvtl.h.b", "$md, $ms1">;
def TH_MSFCVTH_H_B : THRVMInstEWConv<0b0001, 0b00, 0b011, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.msfcvth.h.b", "$md, $ms1">;
def TH_MUFCVTL_H_B : THRVMInstEWConv<0b0001, 0b00, 0b000, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mufcvtl.h.b", "$md, $ms1">;
def TH_MUFCVTH_H_B : THRVMInstEWConv<0b0001, 0b00, 0b010, 0b00, 0b01,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mufcvth.h.b", "$md, $ms1">;

// INT32 <-> FP32
def TH_MSFCVT_S_W : THRVMInstEWConv<0b0001, 0b00, 0b001, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.msfcvt.s.w", "$md, $ms1">;
def TH_MUFCVT_S_W : THRVMInstEWConv<0b0001, 0b00, 0b000, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mufcvt.s.w", "$md, $ms1">;
def TH_MFSCVT_W_S : THRVMInstEWConv<0b0001, 0b00, 0b101, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfscvt.w.s", "$md, $ms1">;
def TH_MFUCVT_W_S : THRVMInstEWConv<0b0001, 0b00, 0b100, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfucvt.w.s", "$md, $ms1">;

// FP16 to INT8
def TH_MFUCVTL_B_H : THRVMInstEWConv<0b0001, 0b00, 0b100, 0b01, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfucvtl.b.h", "$md, $ms1">;
def TH_MFUCVTH_B_H : THRVMInstEWConv<0b0001, 0b00, 0b110, 0b01, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfucvth.b.h", "$md, $ms1">;
def TH_MFSCVTL_B_H : THRVMInstEWConv<0b0001, 0b00, 0b101, 0b01, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfscvtl.b.h", "$md, $ms1">;
def TH_MFSCVTH_B_H : THRVMInstEWConv<0b0001, 0b00, 0b111, 0b01, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mfscvth.b.h", "$md, $ms1">;

//--- Fixed-point N4Clip (uop=00, func4=0010-0101) ---

def TH_MN4CLIPL_W_MM : THRVMInstEWBinMM<0b0010, 0b00, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mn4clipl.w.mm", "$md, $ms2, $ms1">;
def TH_MN4CLIPL_W_MV_I : THRVMInstEWBinMVI<0b0010, 0b00, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mn4clipl.w.mv.i", "$md, $ms2, $ms1, $imm3">;

def TH_MN4CLIPH_W_MM : THRVMInstEWBinMM<0b0011, 0b00, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mn4cliph.w.mm", "$md, $ms2, $ms1">;
def TH_MN4CLIPH_W_MV_I : THRVMInstEWBinMVI<0b0011, 0b00, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mn4cliph.w.mv.i", "$md, $ms2, $ms1, $imm3">;

def TH_MN4CLIPLU_W_MM : THRVMInstEWBinMM<0b0100, 0b00, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mn4cliplu.w.mm", "$md, $ms2, $ms1">;
def TH_MN4CLIPLU_W_MV_I : THRVMInstEWBinMVI<0b0100, 0b00, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mn4cliplu.w.mv.i", "$md, $ms2, $ms1, $imm3">;

def TH_MN4CLIPHU_W_MM : THRVMInstEWBinMM<0b0101, 0b00, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
    "th.mn4cliphu.w.mm", "$md, $ms2, $ms1">;
def TH_MN4CLIPHU_W_MV_I : THRVMInstEWBinMVI<0b0101, 0b00, 0b10, 0b10,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1, thrvmuimm3:$imm3),
    "th.mn4cliphu.w.mv.i", "$md, $ms2, $ms1, $imm3">;

//--- Packed conversions (uop=00, func4=0110) ---

def TH_MUCVTL_B_P : THRVMInstEWConv<0b0110, 0b00, 0b000, 0b00, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mucvtl.b.p", "$md, $ms1">;
def TH_MSCVTL_B_P : THRVMInstEWConv<0b0110, 0b00, 0b001, 0b00, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mscvtl.b.p", "$md, $ms1">;
def TH_MUCVTH_B_P : THRVMInstEWConv<0b0110, 0b00, 0b010, 0b00, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mucvth.b.p", "$md, $ms1">;
def TH_MSCVTH_B_P : THRVMInstEWConv<0b0110, 0b00, 0b011, 0b00, 0b00,
    (outs), (ins THRVMMR:$md, THRVMMR:$ms1),
    "th.mscvth.b.p", "$md, $ms1">;

//--- Integer element-wise arithmetic (uop=01, func3=001) ---

multiclass THRVMEWIntArith<string mnemonic, bits<4> func4> {
  def _W_MM : THRVMInstEWBinMM<func4, 0b01, 0b10, 0b10,
      (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
      "th."#mnemonic#".w.mm", "$md, $ms2, $ms1">;
  def _W_MV_I : THRVMInstEWBinMVI<func4, 0b01, 0b10, 0b10,
      (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1, thrvmuimm3:$imm3),
      "th."#mnemonic#".w.mv.i", "$md, $ms2, $ms1, $imm3">;
}

defm TH_MADD   : THRVMEWIntArith<"madd",   0b0000>;
defm TH_MSUB   : THRVMEWIntArith<"msub",   0b0001>;
defm TH_MMUL   : THRVMEWIntArith<"mmul",   0b0010>;
defm TH_MMULH  : THRVMEWIntArith<"mmulh",  0b0011>;
defm TH_MMAX   : THRVMEWIntArith<"mmax",   0b0100>;
defm TH_MUMAX  : THRVMEWIntArith<"mumax",  0b0101>;
defm TH_MMIN   : THRVMEWIntArith<"mmin",   0b0110>;
defm TH_MUMIN  : THRVMEWIntArith<"mumin",  0b0111>;
defm TH_MSRL   : THRVMEWIntArith<"msrl",   0b1000>;
defm TH_MSLL   : THRVMEWIntArith<"msll",   0b1001>;
defm TH_MSRA   : THRVMEWIntArith<"msra",   0b1010>;

//--- FP element-wise arithmetic (uop=10, func3=001) ---

multiclass THRVMEWFPArith<string mnemonic, bits<4> func4> {
  def _H_MM : THRVMInstEWBinMM<func4, 0b10, 0b01, 0b01,
      (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
      "th."#mnemonic#".h.mm", "$md, $ms2, $ms1">;
  def _H_MV_I : THRVMInstEWBinMVI<func4, 0b10, 0b01, 0b01,
      (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1, thrvmuimm3:$imm3),
      "th."#mnemonic#".h.mv.i", "$md, $ms2, $ms1, $imm3">;
  def _S_MM : THRVMInstEWBinMM<func4, 0b10, 0b10, 0b10,
      (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
      "th."#mnemonic#".s.mm", "$md, $ms2, $ms1">;
  def _S_MV_I : THRVMInstEWBinMVI<func4, 0b10, 0b10, 0b10,
      (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1, thrvmuimm3:$imm3),
      "th."#mnemonic#".s.mv.i", "$md, $ms2, $ms1, $imm3">;
  def _D_MM : THRVMInstEWBinMM<func4, 0b10, 0b11, 0b11,
      (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1),
      "th."#mnemonic#".d.mm", "$md, $ms2, $ms1">;
  def _D_MV_I : THRVMInstEWBinMVI<func4, 0b10, 0b11, 0b11,
      (outs), (ins THRVMMR:$md, THRVMMR:$ms2, THRVMMR:$ms1, thrvmuimm3:$imm3),
      "th."#mnemonic#".d.mv.i", "$md, $ms2, $ms1, $imm3">;
}

defm TH_MFADD : THRVMEWFPArith<"mfadd", 0b0000>;
defm TH_MFSUB : THRVMEWFPArith<"mfsub", 0b0001>;
defm TH_MFMUL : THRVMEWFPArith<"mfmul", 0b0010>;
defm TH_MFMAX : THRVMEWFPArith<"mfmax", 0b0011>;
defm TH_MFMIN : THRVMEWFPArith<"mfmin", 0b0100>;

} // hasSideEffects = 1

//===----------------------------------------------------------------------===//
// ManagedRA Pseudo-Instructions
//===----------------------------------------------------------------------===//
// These pseudo-instructions have proper (outs) register class constraints for
// use with the register allocator. They are expanded to real instructions
// post-RA by RISCVExpandPseudoInsts.

let hasSideEffects = 1, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1 in {

// --- Load A (tile-register output) ---
foreach eew = [8, 16, 32, 64] in {
  def PTH_MLAE_E # eew # V
      : Pseudo<(outs THRVMTR:$dst), (ins GPR:$base, GPR:$stride), []>;
  def PTH_MLATE_E # eew # V
      : Pseudo<(outs THRVMTR:$dst), (ins GPR:$base, GPR:$stride), []>;
}

// --- Load B (tile-register output) ---
foreach eew = [8, 16, 32, 64] in {
  def PTH_MLBE_E # eew # V
      : Pseudo<(outs THRVMTR:$dst), (ins GPR:$base, GPR:$stride), []>;
  def PTH_MLBTE_E # eew # V
      : Pseudo<(outs THRVMTR:$dst), (ins GPR:$base, GPR:$stride), []>;
}

// --- Load C (accumulator output) ---
foreach eew = [8, 16, 32, 64] in {
  def PTH_MLCE_E # eew # V
      : Pseudo<(outs THRVMACC:$dst), (ins GPR:$base, GPR:$stride), []>;
  def PTH_MLCTE_E # eew # V
      : Pseudo<(outs THRVMACC:$dst), (ins GPR:$base, GPR:$stride), []>;
}

// --- Load M (whole-register, any matrix register) ---
foreach eew = [8, 16, 32, 64] in {
  def PTH_MLME_E # eew # V
      : Pseudo<(outs THRVMMR:$dst), (ins GPR:$base), []>;
}

} // hasSideEffects = 1, mayLoad = 1

let hasSideEffects = 1, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1 in {

// --- Store A (tile-register input) ---
foreach eew = [8, 16, 32, 64] in {
  def PTH_MSAE_E # eew # V
      : Pseudo<(outs), (ins THRVMTR:$src, GPR:$base, GPR:$stride), []>;
  def PTH_MSATE_E # eew # V
      : Pseudo<(outs), (ins THRVMTR:$src, GPR:$base, GPR:$stride), []>;
}

// --- Store B (tile-register input) ---
foreach eew = [8, 16, 32, 64] in {
  def PTH_MSBE_E # eew # V
      : Pseudo<(outs), (ins THRVMTR:$src, GPR:$base, GPR:$stride), []>;
  def PTH_MSBTE_E # eew # V
      : Pseudo<(outs), (ins THRVMTR:$src, GPR:$base, GPR:$stride), []>;
}

// --- Store C (accumulator input) ---
foreach eew = [8, 16, 32, 64] in {
  def PTH_MSCE_E # eew # V
      : Pseudo<(outs), (ins THRVMACC:$src, GPR:$base, GPR:$stride), []>;
  def PTH_MSCTE_E # eew # V
      : Pseudo<(outs), (ins THRVMACC:$src, GPR:$base, GPR:$stride), []>;
}

// --- Store M (whole-register) ---
foreach eew = [8, 16, 32, 64] in {
  def PTH_MSME_E # eew # V
      : Pseudo<(outs), (ins THRVMMR:$src, GPR:$base), []>;
}

} // hasSideEffects = 1, mayStore = 1

let hasSideEffects = 1, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1 in {

// --- Matmul (accumulate): dst/acc = acc + f(ms2, ms1) ---
let Constraints = "$acc = $dst" in {
def PTH_MFMACC_H_V       : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_S_V       : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_D_V       : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_H_E4_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_H_E5_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_BF16_E4_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_BF16_E5_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_S_H_V     : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_S_BF16_V  : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_S_E4_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_S_E5_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_S_TF32_V  : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MFMACC_D_S_V     : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MMACC_W_B_V      : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MMACCU_W_B_V     : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MMACCUS_W_B_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MMACCSU_W_B_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MMACC_D_H_V      : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MMACCU_D_H_V     : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MMACCUS_D_H_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MMACCSU_D_H_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_PMMACC_W_B_V     : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_PMMACCU_W_B_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_PMMACCUS_W_B_V   : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_PMMACCSU_W_B_V   : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MMACC_W_BP_V     : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
def PTH_MMACCU_W_BP_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$acc, THRVMTR:$ms2, THRVMTR:$ms1), []>;
} // Constraints = "$acc = $dst"

// --- Zero (produces matrix value, no input) ---
def PTH_MZERO_V   : Pseudo<(outs THRVMMR:$dst), (ins), []>;
def PTH_MZERO2R_V : Pseudo<(outs THRVMMR:$dst), (ins), []>;
def PTH_MZERO4R_V : Pseudo<(outs THRVMMR:$dst), (ins), []>;
def PTH_MZERO8R_V : Pseudo<(outs THRVMMR:$dst), (ins), []>;

// --- Move matrix-to-matrix ---
def PTH_MMOV_MM_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms1), []>;

// --- Move matrix-element-to-GPR ---
def PTH_MMOVB_X_M_V : Pseudo<(outs GPR:$rd), (ins THRVMMR:$ms2, GPR:$rs1), []>;
def PTH_MMOVH_X_M_V : Pseudo<(outs GPR:$rd), (ins THRVMMR:$ms2, GPR:$rs1), []>;
def PTH_MMOVW_X_M_V : Pseudo<(outs GPR:$rd), (ins THRVMMR:$ms2, GPR:$rs1), []>;
def PTH_MMOVD_X_M_V : Pseudo<(outs GPR:$rd), (ins THRVMMR:$ms2, GPR:$rs1), []>;

// --- Move GPR-to-matrix-element (modify, tied) ---
let Constraints = "$md_in = $dst" in {
def PTH_MMOVB_M_X_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$md_in, GPR:$rs2, GPR:$rs1), []>;
def PTH_MMOVH_M_X_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$md_in, GPR:$rs2, GPR:$rs1), []>;
def PTH_MMOVW_M_X_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$md_in, GPR:$rs2, GPR:$rs1), []>;
def PTH_MMOVD_M_X_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$md_in, GPR:$rs2, GPR:$rs1), []>;
}

// --- Dup GPR to matrix column (modify, tied) ---
let Constraints = "$md_in = $dst" in {
def PTH_MDUPB_M_X_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$md_in, GPR:$rs2), []>;
def PTH_MDUPH_M_X_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$md_in, GPR:$rs2), []>;
def PTH_MDUPW_M_X_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$md_in, GPR:$rs2), []>;
def PTH_MDUPD_M_X_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$md_in, GPR:$rs2), []>;
}

// --- Pack (no accumulate) ---
def PTH_MPACK_V   : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms2, THRVMMR:$ms1), []>;
def PTH_MPACKHL_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms2, THRVMMR:$ms1), []>;
def PTH_MPACKHH_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms2, THRVMMR:$ms1), []>;

// --- Row slide ---
def PTH_MRSLIDEDOWN_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms1, thrvmuimm3:$imm3), []>;
def PTH_MRSLIDEUP_V   : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms1, thrvmuimm3:$imm3), []>;

// --- Column slide ---
foreach sz = ["B", "H", "W", "D"] in {
  def PTH_MCSLIDEDOWN_ # sz # _V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms1, thrvmuimm3:$imm3), []>;
  def PTH_MCSLIDEUP_ # sz # _V   : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms1, thrvmuimm3:$imm3), []>;
}

// --- Row broadcast ---
def PTH_MRBCA_MV_I_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms1, thrvmuimm3:$imm3), []>;

// --- Column broadcast ---
def PTH_MCBCAB_MV_I_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms1, thrvmuimm3:$imm3), []>;
def PTH_MCBCAH_MV_I_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms1, thrvmuimm3:$imm3), []>;
def PTH_MCBCAW_MV_I_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms1, thrvmuimm3:$imm3), []>;
def PTH_MCBCAD_MV_I_V : Pseudo<(outs THRVMMR:$dst), (ins THRVMMR:$ms1, thrvmuimm3:$imm3), []>;

// --- Conversions (unary: ms1 -> md) ---
// Per RVM 0.6 spec: "The input and output matrices of type-convert
// instructions are both accumulation registers."
def PTH_MFCVTL_H_E4_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_H_E4_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_H_E5_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_H_E5_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_E4_H_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_E4_H_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_E5_H_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_E5_H_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_S_H_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_S_H_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_H_S_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_H_S_V    : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_S_BF16_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_S_BF16_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_BF16_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_BF16_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_E4_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_E4_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_E5_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_E5_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_D_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_D_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTL_S_D_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVTH_S_D_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVT_S_TF32_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFCVT_TF32_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;

// Float-int conversions
def PTH_MUFCVTL_H_B_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MUFCVTH_H_B_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MSFCVTL_H_B_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MSFCVTH_H_B_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFUCVTL_B_H_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFUCVTH_B_H_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFSCVTL_B_H_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFSCVTH_B_H_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MSFCVT_S_W_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MUFCVT_S_W_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFSCVT_W_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MFUCVT_W_S_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;

// Packed conversions
def PTH_MUCVTL_B_P_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MSCVTL_B_P_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MUCVTH_B_P_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;
def PTH_MSCVTH_B_P_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$ms1), []>;

// --- N4clip .mm (accumulate, tied) ---
let Constraints = "$src1 = $dst" in {
def PTH_MN4CLIPL_W_MM_V  : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1), []>;
def PTH_MN4CLIPH_W_MM_V  : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1), []>;
def PTH_MN4CLIPLU_W_MM_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1), []>;
def PTH_MN4CLIPHU_W_MM_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1), []>;
}

// --- N4clip .mv.i (accumulate + imm, tied) ---
let Constraints = "$src1 = $dst" in {
def PTH_MN4CLIPL_W_MV_I_V  : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1, thrvmuimm3:$imm3), []>;
def PTH_MN4CLIPH_W_MV_I_V  : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1, thrvmuimm3:$imm3), []>;
def PTH_MN4CLIPLU_W_MV_I_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1, thrvmuimm3:$imm3), []>;
def PTH_MN4CLIPHU_W_MV_I_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1, thrvmuimm3:$imm3), []>;
}

// --- Integer EW .mm (accumulate, tied) ---
let Constraints = "$src1 = $dst" in {
foreach op = ["MADD", "MSUB", "MMUL", "MMULH",
              "MMAX", "MUMAX", "MMIN", "MUMIN",
              "MSRL", "MSLL", "MSRA"] in
  def PTH_ # op # _W_MM_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1), []>;
}

// --- Integer EW .mv.i (accumulate + imm, tied) ---
let Constraints = "$src1 = $dst" in {
foreach op = ["MADD", "MSUB", "MMUL", "MMULH",
              "MMAX", "MUMAX", "MMIN", "MUMIN",
              "MSRL", "MSLL", "MSRA"] in
  def PTH_ # op # _W_MV_I_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1, thrvmuimm3:$imm3), []>;
}

// --- FP EW .mm (accumulate, tied) ---
let Constraints = "$src1 = $dst" in {
foreach op = ["MFADD", "MFSUB", "MFMUL", "MFMAX", "MFMIN"] in {
  foreach sz = ["H", "S", "D"] in
    def PTH_ # op # _ # sz # _MM_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1), []>;
}
}

// --- FP EW .mv.i (accumulate + imm, tied) ---
let Constraints = "$src1 = $dst" in {
foreach op = ["MFADD", "MFSUB", "MFMUL", "MFMAX", "MFMIN"] in {
  foreach sz = ["H", "S", "D"] in
    def PTH_ # op # _ # sz # _MV_I_V : Pseudo<(outs THRVMACC:$dst), (ins THRVMACC:$src1, THRVMACC:$ms2, THRVMACC:$ms1, thrvmuimm3:$imm3), []>;
}
}

} // hasSideEffects = 1, mayLoad = 0, mayStore = 0

// --- Spill/Reload pseudos for register allocator ---
// These use the RVV-style spill pattern: no immediate offset operand.
// eliminateFrameIndex materializes the full address into the GPR operand.
let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1 in
def PTH_MATRIX_SPILL : Pseudo<(outs), (ins THRVMMR:$src, GPR:$addr), []>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1 in
def PTH_MATRIX_RELOAD : Pseudo<(outs THRVMMR:$dst), (ins GPR:$addr), []>;

} // Predicates = [HasVendorXTHeadMatrix]

//===----------------------------------------------------------------------===//
// Zmpanel: Panel-Aware 2x2 Matrix Tiling Extension
//===----------------------------------------------------------------------===//
// All panel-aware instructions use func3=010 (bits[14:12]) to distinguish
// from standard RVM instructions (func3=000). They operate on implicit
// hardware state and do not have explicit matrix register operands.

// --- Panel-aware format classes ---

// Panel config: func3=010, uop=00, ctrl=0, takes GPR rs1
class THRVMInstPanelCfg<bits<4> func4,
                         dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  bits<5> rs1;

  let Inst{31-28} = func4;
  let Inst{27-26} = 0b00;
  let Inst{25} = 0;
  let Inst{24-20} = 0b00000;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b010;
  let Inst{11-7} = 0b00000;
  let DecoderNamespace = "XTHeadZmpanel";
}

// Panel load: func3=010, uop=01, ls=0, no operands
class THRVMInstPanelLoad<bits<2> dsize,
                          dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  let Inst{31-28} = 0b0000;
  let Inst{27-26} = 0b01;
  let Inst{25} = 0;        // ls=0: load
  let Inst{24-20} = 0b00000;
  let Inst{19-15} = 0b00000;
  let Inst{14-12} = 0b010;
  let Inst{11-10} = dsize;
  let Inst{9-7} = 0b000;
  let mayLoad = 1;
  let mayStore = 0;
  let DecoderNamespace = "XTHeadZmpanel";
}

// Panel store: func3=010, func4=0010, uop=01, ls=1, no operands
class THRVMInstPanelStore<bits<2> dsize,
                           dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  let Inst{31-28} = 0b0010;
  let Inst{27-26} = 0b01;
  let Inst{25} = 1;        // ls=1: store
  let Inst{24-20} = 0b00000;
  let Inst{19-15} = 0b00000;
  let Inst{14-12} = 0b010;
  let Inst{11-10} = dsize;
  let Inst{9-7} = 0b000;
  let mayLoad = 0;
  let mayStore = 1;
  let DecoderNamespace = "XTHeadZmpanel";
}

// Panel compute: func3=010, uop=10, no register operands
// Mirrors standard matmul encoding with func3=010 and ms2/ms1/md all zero
class THRVMInstPanelCompute<bits<4> func4, bits<3> size_sup, bits<2> ssize,
                             bits<2> dsize,
                             dag outs, dag ins, string opcodestr, string argstr>
    : THRVMInst<outs, ins, opcodestr, argstr> {
  let Inst{31-28} = func4;
  let Inst{27-26} = 0b10;
  let Inst{25-23} = size_sup;
  let Inst{22-20} = 0b000;
  let Inst{19-18} = ssize;
  let Inst{17-15} = 0b000;
  let Inst{14-12} = 0b010;
  let Inst{11-10} = dsize;
  let Inst{9-7} = 0b000;
  let mayLoad = 0;
  let mayStore = 0;
  let DecoderNamespace = "XTHeadZmpanel";
}

// --- Panel-aware instruction definitions ---

let Predicates = [HasVendorXTHeadZmpanel] in {
let hasSideEffects = 1 in {

//--- Configuration instructions ---
let mayLoad = 0, mayStore = 0 in {
def TH_MSET22ADRA : THRVMInstPanelCfg<0b0000,
    (outs), (ins GPR:$rs1), "th.mset22adra", "$rs1">;
def TH_MSET22ADRB : THRVMInstPanelCfg<0b0001,
    (outs), (ins GPR:$rs1), "th.mset22adrb", "$rs1">;
def TH_MSET22ADRD : THRVMInstPanelCfg<0b0010,
    (outs), (ins GPR:$rs1), "th.mset22adrd", "$rs1">;
def TH_MSET22RSBA : THRVMInstPanelCfg<0b0011,
    (outs), (ins GPR:$rs1), "th.mset22rsba", "$rs1">;
def TH_MSET22RSBB : THRVMInstPanelCfg<0b0100,
    (outs), (ins GPR:$rs1), "th.mset22rsbb", "$rs1">;
def TH_MSET22RSBD : THRVMInstPanelCfg<0b0101,
    (outs), (ins GPR:$rs1), "th.mset22rsbd", "$rs1">;
def TH_MSET22M    : THRVMInstPanelCfg<0b0110,
    (outs), (ins GPR:$rs1), "th.mset22m", "$rs1">;
def TH_MSET22N    : THRVMInstPanelCfg<0b0111,
    (outs), (ins GPR:$rs1), "th.mset22n", "$rs1">;
def TH_MSET22K    : THRVMInstPanelCfg<0b1000,
    (outs), (ins GPR:$rs1), "th.mset22k", "$rs1">;
def TH_MSETRSTPTR : THRVMInstPanelCfg<0b1001,
    (outs), (ins GPR:$rs1), "th.msetrstptr", "$rs1">;
def TH_MSETACCUM  : THRVMInstPanelCfg<0b1010,
    (outs), (ins GPR:$rs1), "th.msetaccum", "$rs1">;
def TH_MSETOOB    : THRVMInstPanelCfg<0b1011,
    (outs), (ins GPR:$rs1), "th.msetoob", "$rs1">;
} // mayLoad = 0, mayStore = 0

//--- Load macro instructions ---
// Panel loads clobber all 4 compiler-visible tile registers (tr0-tr3).
// They also write tr4-tr7 but those are not in the compiler's register file.
let Defs = [THRVM_TR0, THRVM_TR1, THRVM_TR2, THRVM_TR3] in {
def TH_ML22E8  : THRVMInstPanelLoad<0b00,
    (outs), (ins), "th.ml22e8", "">;
def TH_ML22E16 : THRVMInstPanelLoad<0b01,
    (outs), (ins), "th.ml22e16", "">;
}

//--- Store macro instructions ---
// Panel stores read all 4 accumulator registers (acc0-acc3).
let Uses = [THRVM_ACC0, THRVM_ACC1, THRVM_ACC2, THRVM_ACC3] in {
def TH_MSC22E16 : THRVMInstPanelStore<0b01,
    (outs), (ins), "th.msc22e16", "">;
def TH_MSC22E32 : THRVMInstPanelStore<0b10,
    (outs), (ins), "th.msc22e32", "">;
}

//--- FP compute macro instructions (func4=0000) ---
// Panel computes read tr0-tr3 (tile inputs) and write acc0-acc3 (results).
// In accumulate mode they also read acc0-acc3, so we mark them as both
// Defs and Uses to be conservative.
let Defs = [THRVM_ACC0, THRVM_ACC1, THRVM_ACC2, THRVM_ACC3],
    Uses = [THRVM_TR0, THRVM_TR1, THRVM_TR2, THRVM_TR3,
            THRVM_ACC0, THRVM_ACC1, THRVM_ACC2, THRVM_ACC3] in {
def TH_MFMACC22_H_E5    : THRVMInstPanelCompute<0b0000, 0b000, 0b00, 0b01,
    (outs), (ins), "th.mfmacc22.h.e5", "">;
def TH_MFMACC22_H_E4    : THRVMInstPanelCompute<0b0000, 0b001, 0b00, 0b01,
    (outs), (ins), "th.mfmacc22.h.e4", "">;
def TH_MFMACC22_BF16_E5 : THRVMInstPanelCompute<0b0000, 0b100, 0b00, 0b01,
    (outs), (ins), "th.mfmacc22.bf16.e5", "">;
def TH_MFMACC22_BF16_E4 : THRVMInstPanelCompute<0b0000, 0b101, 0b00, 0b01,
    (outs), (ins), "th.mfmacc22.bf16.e4", "">;
def TH_MFMACC22_S_E5    : THRVMInstPanelCompute<0b0000, 0b000, 0b00, 0b10,
    (outs), (ins), "th.mfmacc22.s.e5", "">;
def TH_MFMACC22_S_E4    : THRVMInstPanelCompute<0b0000, 0b001, 0b00, 0b10,
    (outs), (ins), "th.mfmacc22.s.e4", "">;
def TH_MFMACC22_H       : THRVMInstPanelCompute<0b0000, 0b000, 0b01, 0b01,
    (outs), (ins), "th.mfmacc22.h", "">;
def TH_MFMACC22_S_H     : THRVMInstPanelCompute<0b0000, 0b000, 0b01, 0b10,
    (outs), (ins), "th.mfmacc22.s.h", "">;
def TH_MFMACC22_S_BF16  : THRVMInstPanelCompute<0b0000, 0b001, 0b01, 0b10,
    (outs), (ins), "th.mfmacc22.s.bf16", "">;
def TH_MFMACC22_S       : THRVMInstPanelCompute<0b0000, 0b000, 0b10, 0b10,
    (outs), (ins), "th.mfmacc22.s", "">;

//--- INT compute macro instructions (func4=0001) ---
def TH_MMACC22_W_B      : THRVMInstPanelCompute<0b0001, 0b011, 0b00, 0b10,
    (outs), (ins), "th.mmacc22.w.b", "">;
def TH_MMACCU22_W_B     : THRVMInstPanelCompute<0b0001, 0b000, 0b00, 0b10,
    (outs), (ins), "th.mmaccu22.w.b", "">;
def TH_MMACCUS22_W_B    : THRVMInstPanelCompute<0b0001, 0b001, 0b00, 0b10,
    (outs), (ins), "th.mmaccus22.w.b", "">;
def TH_MMACCSU22_W_B    : THRVMInstPanelCompute<0b0001, 0b010, 0b00, 0b10,
    (outs), (ins), "th.mmaccsu22.w.b", "">;
} // Defs, Uses

} // hasSideEffects = 1
} // Predicates = [HasVendorXTHeadZmpanel]
