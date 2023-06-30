//===- XtensaHIFIInstrInfo.td - Instruction definitions for Xtensa HIFI -*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Tablegen instruction definitiona for Xtensa HIFI extension
//
//===----------------------------------------------------------------------===//

class AE_ABS16S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_abs16s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ABS16S : AE_ABS16S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_abs16s AE_DR:$ae_arth_v1))]>;

class AE_ABS24S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_abs24s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ABS24S : AE_ABS24S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_abs24s AE_DR:$ae_arth_v1))]>;

class AE_ABS32_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_abs32 $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{25} = 1;
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_ABS32 : AE_ABS32_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_abs32 AE_DR:$ae_arth_v1))]>;

class AE_ABS32S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_abs32s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ABS32S : AE_ABS32S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_abs32s AE_DR:$ae_arth_v1))]>;

class AE_ABS64_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_abs64 $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{11} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ABS64 : AE_ABS64_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_abs64 AE_DR:$ae_arth_v1))]>;

class AE_ABS64S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_abs64s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{24} = 1;
let Inst{26} = 1;
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_ABS64S : AE_ABS64S_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_abs64s AE_DR:$ae_arth_v1))]>;

class AE_ADD16_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_add16 $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{50} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{24} = ae_arth_v0{0};
let Inst{25} = ae_arth_v0{1};
let Inst{26} = ae_arth_v0{2};
let Inst{27} = ae_arth_v0{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_ADD16 : AE_ADD16_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_add16 AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ADD16S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_add16s $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ADD16S : AE_ADD16S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_add16s AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ADD24S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_add24s $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ADD24S : AE_ADD24S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_add24s AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ADD32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_add32 $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ADD32 : AE_ADD32_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_add32 AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ADD32_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_add32_hl_lh $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{51} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{24} = ae_arth_v0{0};
let Inst{25} = ae_arth_v0{1};
let Inst{26} = ae_arth_v0{2};
let Inst{27} = ae_arth_v0{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_ADD32_HL_LH : AE_ADD32_HL_LH_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_add32_hl_lh AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ADD32S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_add32s $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ADD32S : AE_ADD32S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_add32s AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ADD64_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_add64 $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ADD64 : AE_ADD64_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_add64 AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ADD64S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_add64s $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{24} = ae_arth_v0{0};
let Inst{25} = ae_arth_v0{1};
let Inst{26} = ae_arth_v0{2};
let Inst{27} = ae_arth_v0{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_ADD64S : AE_ADD64S_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_add64s AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ADDBRBA32_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$arr), (ins AR:$art, AR:$ars), "ae_addbrba32 $arr, $art, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> art;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{20} = arr{0};
let Inst{21} = arr{1};
let Inst{22} = arr{2};
let Inst{23} = arr{3};
let Inst{24} = art{0};
let Inst{25} = art{1};
let Inst{26} = art{2};
let Inst{27} = art{3};
let Inst{16} = ars{0};
let Inst{17} = ars{1};
let Inst{18} = ars{2};
let Inst{19} = ars{3};
}



def AE_ADDBRBA32 : AE_ADDBRBA32_AE_FORMAT48<[(set AR:$arr, (int_xtensa_ae_addbrba32 AR:$art, AR:$ars))]>;

class AE_ADDSUB32_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_addsub32 $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{24} = ae_arth_v0{0};
let Inst{25} = ae_arth_v0{1};
let Inst{26} = ae_arth_v0{2};
let Inst{27} = ae_arth_v0{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_ADDSUB32 : AE_ADDSUB32_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_addsub32 AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ADDSUB32S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_addsub32s $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{24} = ae_arth_v0{0};
let Inst{25} = ae_arth_v0{1};
let Inst{26} = ae_arth_v0{2};
let Inst{27} = ae_arth_v0{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_ADDSUB32S : AE_ADDSUB32S_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_addsub32s AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_AND_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_dr_to_dr_v), (ins AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1), "ae_and $ae_dr_to_dr_v, $ae_dr_to_dr_v0, $ae_dr_to_dr_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_dr_to_dr_v;
bits<4> ae_dr_to_dr_v0;
bits<4> ae_dr_to_dr_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_dr_to_dr_v{0};
let Inst{13} = ae_dr_to_dr_v{1};
let Inst{14} = ae_dr_to_dr_v{2};
let Inst{15} = ae_dr_to_dr_v{3};
let Inst{8} = ae_dr_to_dr_v0{0};
let Inst{9} = ae_dr_to_dr_v0{1};
let Inst{10} = ae_dr_to_dr_v0{2};
let Inst{11} = ae_dr_to_dr_v0{3};
let Inst{4} = ae_dr_to_dr_v1{0};
let Inst{5} = ae_dr_to_dr_v1{1};
let Inst{6} = ae_dr_to_dr_v1{2};
let Inst{7} = ae_dr_to_dr_v1{3};
}



def AE_AND : AE_AND_X24<[(set AE_DR:$ae_dr_to_dr_v, (int_xtensa_ae_and AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1))]>;

class AE_CVT32X2F16_10_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_to_dr_v), (ins AE_DR:$ae_to_dr_v0), "ae_cvt32x2f16.10 $ae_to_dr_v, $ae_to_dr_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_to_dr_v;
bits<4> ae_to_dr_v0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//operands
let Inst{12} = ae_to_dr_v{0};
let Inst{13} = ae_to_dr_v{1};
let Inst{14} = ae_to_dr_v{2};
let Inst{15} = ae_to_dr_v{3};
let Inst{8} = ae_to_dr_v0{0};
let Inst{9} = ae_to_dr_v0{1};
let Inst{10} = ae_to_dr_v0{2};
let Inst{11} = ae_to_dr_v0{3};
}



def AE_CVT32X2F16_10 : AE_CVT32X2F16_10_AE_FORMAT<[(set AE_DR:$ae_to_dr_v, (int_xtensa_ae_cvt32x2f16_10 AE_DR:$ae_to_dr_v0))]>;

class AE_CVT32X2F16_32_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_to_dr_v), (ins AE_DR:$ae_to_dr_v0), "ae_cvt32x2f16.32 $ae_to_dr_v, $ae_to_dr_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_to_dr_v;
bits<4> ae_to_dr_v0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//operands
let Inst{12} = ae_to_dr_v{0};
let Inst{13} = ae_to_dr_v{1};
let Inst{14} = ae_to_dr_v{2};
let Inst{15} = ae_to_dr_v{3};
let Inst{8} = ae_to_dr_v0{0};
let Inst{9} = ae_to_dr_v0{1};
let Inst{10} = ae_to_dr_v0{2};
let Inst{11} = ae_to_dr_v0{3};
}



def AE_CVT32X2F16_32 : AE_CVT32X2F16_32_AE_FORMAT<[(set AE_DR:$ae_to_dr_v, (int_xtensa_ae_cvt32x2f16_32 AE_DR:$ae_to_dr_v0))]>;

class AE_CVT48A32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ar_to_dr_v), (ins AR:$ars), "ae_cvt48a32 $ae_ar_to_dr_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ar_to_dr_v;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ar_to_dr_v{0};
let Inst{13} = ae_ar_to_dr_v{1};
let Inst{14} = ae_ar_to_dr_v{2};
let Inst{15} = ae_ar_to_dr_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_CVT48A32 : AE_CVT48A32_X24<[(set AE_DR:$ae_ar_to_dr_v, (int_xtensa_ae_cvt48a32 AR:$ars))]>;

class AE_CVT64A32_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ar_to_dr_v), (ins AR:$ars), "ae_cvt64a32 $ae_ar_to_dr_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ar_to_dr_v;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{25} = 1;
let Inst{27} = 1;
let Inst{38} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{20} = ae_ar_to_dr_v{0};
let Inst{21} = ae_ar_to_dr_v{1};
let Inst{22} = ae_ar_to_dr_v{2};
let Inst{23} = ae_ar_to_dr_v{3};
let Inst{16} = ars{0};
let Inst{17} = ars{1};
let Inst{18} = ars{2};
let Inst{19} = ars{3};
}



def AE_CVT64A32 : AE_CVT64A32_AE_FORMAT48<[(set AE_DR:$ae_ar_to_dr_v, (int_xtensa_ae_cvt64a32 AR:$ars))]>;

class AE_CVT64F32_H_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_dr_to_dr_v), (ins AE_DR:$ae_dr_to_dr_v0), "ae_cvt64f32.h $ae_dr_to_dr_v, $ae_dr_to_dr_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_dr_to_dr_v;
bits<4> ae_dr_to_dr_v0;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{28} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_dr_to_dr_v{0};
let Inst{21} = ae_dr_to_dr_v{1};
let Inst{22} = ae_dr_to_dr_v{2};
let Inst{23} = ae_dr_to_dr_v{3};
let Inst{36} = ae_dr_to_dr_v0{0};
let Inst{37} = ae_dr_to_dr_v0{1};
let Inst{38} = ae_dr_to_dr_v0{2};
let Inst{39} = ae_dr_to_dr_v0{3};
}



def AE_CVT64F32_H : AE_CVT64F32_H_AE_FORMAT1<[(set AE_DR:$ae_dr_to_dr_v, (int_xtensa_ae_cvt64f32_h AE_DR:$ae_dr_to_dr_v0))]>;

class AE_CVTA32F24S_H_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_cvta32f24s.h $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{5} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ae_dr_to_ar_v0{0};
let Inst{9} = ae_dr_to_ar_v0{1};
let Inst{10} = ae_dr_to_ar_v0{2};
let Inst{11} = ae_dr_to_ar_v0{3};
}



def AE_CVTA32F24S_H : AE_CVTA32F24S_H_AE_FORMAT<[(set AR:$arr, (int_xtensa_ae_cvta32f24s_h AE_DR:$ae_dr_to_ar_v0))]>;

class AE_CVTA32F24S_L_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_cvta32f24s.l $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ae_dr_to_ar_v0{0};
let Inst{9} = ae_dr_to_ar_v0{1};
let Inst{10} = ae_dr_to_ar_v0{2};
let Inst{11} = ae_dr_to_ar_v0{3};
}



def AE_CVTA32F24S_L : AE_CVTA32F24S_L_AE_FORMAT<[(set AR:$arr, (int_xtensa_ae_cvta32f24s_l AE_DR:$ae_dr_to_ar_v0))]>;

class AE_CVTQ56A32S_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ar_to_dr_v), (ins AR:$ars), "ae_cvtq56a32s $ae_ar_to_dr_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ar_to_dr_v;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{24} = 1;
let Inst{25} = 1;
let Inst{27} = 1;
let Inst{38} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{20} = ae_ar_to_dr_v{0};
let Inst{21} = ae_ar_to_dr_v{1};
let Inst{22} = ae_ar_to_dr_v{2};
let Inst{23} = ae_ar_to_dr_v{3};
let Inst{16} = ars{0};
let Inst{17} = ars{1};
let Inst{18} = ars{2};
let Inst{19} = ars{3};
}



def AE_CVTQ56A32S : AE_CVTQ56A32S_AE_FORMAT48<[(set AE_DR:$ae_ar_to_dr_v, (int_xtensa_ae_cvtq56a32s AR:$ars))]>;

class AE_CVTQ56P32S_H_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_dr_to_dr_v), (ins AE_DR:$ae_dr_to_dr_v0), "ae_cvtq56p32s.h $ae_dr_to_dr_v, $ae_dr_to_dr_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_dr_to_dr_v;
bits<4> ae_dr_to_dr_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{5} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_dr_to_dr_v{0};
let Inst{13} = ae_dr_to_dr_v{1};
let Inst{14} = ae_dr_to_dr_v{2};
let Inst{15} = ae_dr_to_dr_v{3};
let Inst{8} = ae_dr_to_dr_v0{0};
let Inst{9} = ae_dr_to_dr_v0{1};
let Inst{10} = ae_dr_to_dr_v0{2};
let Inst{11} = ae_dr_to_dr_v0{3};
}



def AE_CVTQ56P32S_H : AE_CVTQ56P32S_H_X24<[(set AE_DR:$ae_dr_to_dr_v, (int_xtensa_ae_cvtq56p32s_h AE_DR:$ae_dr_to_dr_v0))]>;

class AE_CVTQ56P32S_L_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_dr_to_dr_v), (ins AE_DR:$ae_dr_to_dr_v0), "ae_cvtq56p32s.l $ae_dr_to_dr_v, $ae_dr_to_dr_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_dr_to_dr_v;
bits<4> ae_dr_to_dr_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_dr_to_dr_v{0};
let Inst{13} = ae_dr_to_dr_v{1};
let Inst{14} = ae_dr_to_dr_v{2};
let Inst{15} = ae_dr_to_dr_v{3};
let Inst{8} = ae_dr_to_dr_v0{0};
let Inst{9} = ae_dr_to_dr_v0{1};
let Inst{10} = ae_dr_to_dr_v0{2};
let Inst{11} = ae_dr_to_dr_v0{3};
}



def AE_CVTQ56P32S_L : AE_CVTQ56P32S_L_X24<[(set AE_DR:$ae_dr_to_dr_v, (int_xtensa_ae_cvtq56p32s_l AE_DR:$ae_dr_to_dr_v0))]>;

class AE_DB_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, AR:$art), "ae_db $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{14} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_DB : AE_DB_X24<[(set AR:$ars_out, (int_xtensa_ae_db AR:$ars, AR:$art))]>;

class AE_DB_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, AR:$art), "ae_db.ic $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_DB_IC : AE_DB_IC_X24<[(set AR:$ars_out, (int_xtensa_ae_db_ic AR:$ars, AR:$art))]>;

class AE_DB_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, AR:$art), "ae_db.ip $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_DB_IP : AE_DB_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_db_ip AR:$ars, AR:$art))]>;

class AE_DBI_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, imm1_16:$ae_ohba), "ae_dbi $ars, $ae_ohba", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> ae_ohba;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{12} = 1;
let Inst{14} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_ohba{0};
let Inst{5} = ae_ohba{1};
let Inst{6} = ae_ohba{2};
let Inst{7} = ae_ohba{3};
}



def AE_DBI : AE_DBI_X24<[(set AR:$ars_out, (int_xtensa_ae_dbi AR:$ars, timm:$ae_ohba))]>;

class AE_DBI_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, imm1_16:$ae_ohba), "ae_dbi.ic $ars, $ae_ohba", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> ae_ohba;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_ohba{0};
let Inst{5} = ae_ohba{1};
let Inst{6} = ae_ohba{2};
let Inst{7} = ae_ohba{3};
}



def AE_DBI_IC : AE_DBI_IC_X24<[(set AR:$ars_out, (int_xtensa_ae_dbi_ic AR:$ars, timm:$ae_ohba))]>;

class AE_DBI_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, imm1_16:$ae_ohba), "ae_dbi.ip $ars, $ae_ohba", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> ae_ohba;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{12} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_ohba{0};
let Inst{5} = ae_ohba{1};
let Inst{6} = ae_ohba{2};
let Inst{7} = ae_ohba{3};
}



def AE_DBI_IP : AE_DBI_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_dbi_ip AR:$ars, timm:$ae_ohba))]>;

class AE_DIV64D32_H_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v_out), (ins AE_DR:$ae_arth_v, AE_DR:$ae_arth_v1), "ae_div64d32.h $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;
let Constraints = "$ae_arth_v = $ae_arth_v_out,@earlyclobber $ae_arth_v_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{36} = ae_arth_v1{0};
let Inst{37} = ae_arth_v1{1};
let Inst{38} = ae_arth_v1{2};
let Inst{39} = ae_arth_v1{3};
}



def AE_DIV64D32_H : AE_DIV64D32_H_AE_FORMAT1<[(set AE_DR:$ae_arth_v_out, (int_xtensa_ae_div64d32_h AE_DR:$ae_arth_v, AE_DR:$ae_arth_v1))]>;

class AE_DIV64D32_L_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v_out), (ins AE_DR:$ae_arth_v, AE_DR:$ae_arth_v1), "ae_div64d32.l $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;
let Constraints = "$ae_arth_v = $ae_arth_v_out,@earlyclobber $ae_arth_v_out";


//format

//opcode
let Inst{2} = 1;
let Inst{10} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_DIV64D32_L : AE_DIV64D32_L_X24<[(set AE_DR:$ae_arth_v_out, (int_xtensa_ae_div64d32_l AE_DR:$ae_arth_v, AE_DR:$ae_arth_v1))]>;

class AE_EQ16_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs BR4:$br4), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_eq16 $br4, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> br4;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{20} = 1;
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{62} = 1;
//operands
let Inst{22} = br4{0};
let Inst{23} = br4{1};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_EQ16 : AE_EQ16_AE_FORMAT1<[(set BR4:$br4, (int_xtensa_ae_eq16 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_EQ32_X24<list<dag> pattern>
    : XtensaInst24<(outs BR2:$br2), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_eq32 $br2, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<3> br2;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
//operands
let Inst{13} = br2{0};
let Inst{14} = br2{1};
let Inst{15} = br2{2};
let Inst{8} = ae_cmpp_v0{0};
let Inst{9} = ae_cmpp_v0{1};
let Inst{10} = ae_cmpp_v0{2};
let Inst{11} = ae_cmpp_v0{3};
let Inst{4} = ae_cmpp_v1{0};
let Inst{5} = ae_cmpp_v1{1};
let Inst{6} = ae_cmpp_v1{2};
let Inst{7} = ae_cmpp_v1{3};
}



def AE_EQ32 : AE_EQ32_X24<[(set BR2:$br2, (int_xtensa_ae_eq32 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_EQ64_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs BR:$br), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_eq64 $br, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> br;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = br{0};
let Inst{21} = br{1};
let Inst{22} = br{2};
let Inst{23} = br{3};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_EQ64 : AE_EQ64_AE_FORMAT1<[(set BR:$br, (int_xtensa_ae_eq64 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_L16_I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, imm16n_14:$ae_immls16), "ae_l16.i $ae_ls_v, $ars, $ae_immls16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls16;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls16{0};
let Inst{5} = ae_immls16{1};
let Inst{6} = ae_immls16{2};
let Inst{7} = ae_immls16{3};
}



def AE_L16_I : AE_L16_I_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l16_i AR:$ars, timm:$ae_immls16))]>;

class AE_L16_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, imm16n_14:$ae_immls16), "ae_l16.ip $ae_ls_v, $ars, $ae_immls16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls16;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls16{0};
let Inst{5} = ae_immls16{1};
let Inst{6} = ae_immls16{2};
let Inst{7} = ae_immls16{3};
}



def AE_L16_IP : AE_L16_IP_X24<[]>;

class AE_L16_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v), (ins AR:$ars, AR:$art), "ae_l16.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;

let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{30} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16_X : AE_L16_X_AE_FORMAT48<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l16_x AR:$ars, AR:$art))]>;

class AE_L16_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l16.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{30} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16_XC : AE_L16_XC_AE_FORMAT48<[]>;

class AE_L16_XP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l16.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16_XP : AE_L16_XP_AE_FORMAT48<[]>;

class AE_L16M_I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, imm16n_14:$ae_immls16), "ae_l16m.i $ae_ls_v, $ars, $ae_immls16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls16;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls16{0};
let Inst{5} = ae_immls16{1};
let Inst{6} = ae_immls16{2};
let Inst{7} = ae_immls16{3};
}



def AE_L16M_I : AE_L16M_I_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l16m_i AR:$ars, timm:$ae_immls16))]>;

class AE_L16M_IU_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, imm16n_14:$ae_immls16), "ae_l16m.iu $ae_ls_v, $ars, $ae_immls16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls16;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls16{0};
let Inst{5} = ae_immls16{1};
let Inst{6} = ae_immls16{2};
let Inst{7} = ae_immls16{3};
}



def AE_L16M_IU : AE_L16M_IU_X24<[]>;

class AE_L16M_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v), (ins AR:$ars, AR:$art), "ae_l16m.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;

let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16M_X : AE_L16M_X_AE_FORMAT48<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l16m_x AR:$ars, AR:$art))]>;

class AE_L16M_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l16m.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16M_XC : AE_L16M_XC_AE_FORMAT48<[]>;

class AE_L16M_XU_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l16m.xu $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16M_XU : AE_L16M_XU_X24<[]>;

class AE_L16X2M_I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, imm32n_28:$ae_immls32), "ae_l16x2m.i $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_L16X2M_I : AE_L16X2M_I_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l16x2m_i AR:$ars, timm:$ae_immls32))]>;

class AE_L16X2M_IU_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, imm32n_28:$ae_immls32), "ae_l16x2m.iu $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_L16X2M_IU : AE_L16X2M_IU_X24<[]>;

class AE_L16X2M_X_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, AR:$art), "ae_l16x2m.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16X2M_X : AE_L16X2M_X_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l16x2m_x AR:$ars, AR:$art))]>;

class AE_L16X2M_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l16x2m.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16X2M_XC : AE_L16X2M_XC_AE_FORMAT48<[]>;

class AE_L16X2M_XU_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l16x2m.xu $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16X2M_XU : AE_L16X2M_XU_X24<[]>;

class AE_L16X4_I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, imm64n_56:$ae_immls64), "ae_l16x4.i $ae_ls_v, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls64;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{6} = ae_immls64{2};
let Inst{7} = ae_immls64{3};
}



def AE_L16X4_I : AE_L16X4_I_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l16x4_i AR:$ars, timm:$ae_immls64))]>;

class AE_L16X4_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, imm0_56:$ae_immls64pos), "ae_l16x4.ip $ae_ls_v, $ars, $ae_immls64pos", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<3> ae_immls64pos;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{20} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64pos{0};
let Inst{5} = ae_immls64pos{1};
let Inst{6} = ae_immls64pos{2};
}



def AE_L16X4_IP : AE_L16X4_IP_X24<[]>;

class AE_L16X4_RIC_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars), "ae_l16x4.ric $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{5} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_L16X4_RIC : AE_L16X4_RIC_AE_FORMAT<[]>;

class AE_L16X4_RIP_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars), "ae_l16x4.rip $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_L16X4_RIP : AE_L16X4_RIP_AE_FORMAT<[]>;

class AE_L16X4_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v), (ins AR:$ars, AR:$art), "ae_l16x4.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;

let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16X4_X : AE_L16X4_X_AE_FORMAT48<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l16x4_x AR:$ars, AR:$art))]>;

class AE_L16X4_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l16x4.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16X4_XC : AE_L16X4_XC_AE_FORMAT48<[]>;

class AE_L16X4_XP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l16x4.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L16X4_XP : AE_L16X4_XP_X24<[]>;

class AE_L32_I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, imm32n_28:$ae_immls32), "ae_l32.i $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_L32_I : AE_L32_I_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l32_i AR:$ars, timm:$ae_immls32))]>;

class AE_L32_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, imm32n_28:$ae_immls32), "ae_l32.ip $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_L32_IP : AE_L32_IP_X24<[]>;

class AE_L32_X_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, AR:$art), "ae_l32.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32_X : AE_L32_X_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l32_x AR:$ars, AR:$art))]>;

class AE_L32_XC_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l32.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32_XC : AE_L32_XC_X24<[]>;

class AE_L32_XP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l32.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32_XP : AE_L32_XP_AE_FORMAT48<[]>;

class AE_L32F24_I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, imm32n_28:$ae_immls32), "ae_l32f24.i $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_L32F24_I : AE_L32F24_I_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l32f24_i AR:$ars, timm:$ae_immls32))]>;

class AE_L32F24_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, imm32n_28:$ae_immls32), "ae_l32f24.ip $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_L32F24_IP : AE_L32F24_IP_X24<[]>;

class AE_L32F24_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v), (ins AR:$ars, AR:$art), "ae_l32f24.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;

let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32F24_X : AE_L32F24_X_AE_FORMAT48<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l32f24_x AR:$ars, AR:$art))]>;

class AE_L32F24_XC_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l32f24.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32F24_XC : AE_L32F24_XC_X24<[]>;

class AE_L32F24_XP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l32f24.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{31} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32F24_XP : AE_L32F24_XP_AE_FORMAT48<[]>;

class AE_L32M_I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, imm32n_28:$ae_immls32), "ae_l32m.i $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_L32M_I : AE_L32M_I_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l32m_i AR:$ars, timm:$ae_immls32))]>;

class AE_L32M_IU_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, imm32n_28:$ae_immls32), "ae_l32m.iu $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_L32M_IU : AE_L32M_IU_X24<[]>;

class AE_L32M_X_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, AR:$art), "ae_l32m.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32M_X : AE_L32M_X_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l32m_x AR:$ars, AR:$art))]>;

class AE_L32M_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l32m.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{31} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32M_XC : AE_L32M_XC_AE_FORMAT48<[]>;

class AE_L32M_XU_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l32m.xu $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32M_XU : AE_L32M_XU_X24<[]>;

class AE_L32X2_I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, imm64n_56:$ae_immls64), "ae_l32x2.i $ae_ls_v, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls64;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{6} = ae_immls64{2};
let Inst{7} = ae_immls64{3};
}



def AE_L32X2_I : AE_L32X2_I_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l32x2_i AR:$ars, timm:$ae_immls64))]>;

class AE_L32X2_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, imm0_56:$ae_immls64pos), "ae_l32x2.ip $ae_ls_v, $ars, $ae_immls64pos", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<3> ae_immls64pos;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64pos{0};
let Inst{5} = ae_immls64pos{1};
let Inst{6} = ae_immls64pos{2};
}



def AE_L32X2_IP : AE_L32X2_IP_X24<[]>;

class AE_L32X2_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars), "ae_l32x2.ric $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_L32X2_RIC : AE_L32X2_RIC_AE_FORMAT48<[]>;

class AE_L32X2_RIP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars), "ae_l32x2.rip $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_L32X2_RIP : AE_L32X2_RIP_X24<[]>;

class AE_L32X2_X_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, AR:$art), "ae_l32x2.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32X2_X : AE_L32X2_X_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l32x2_x AR:$ars, AR:$art))]>;

class AE_L32X2_XC_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l32x2.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32X2_XC : AE_L32X2_XC_X24<[]>;

class AE_L32X2_XP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l32x2.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32X2_XP : AE_L32X2_XP_X24<[]>;

class AE_L32X2F24_I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, imm64n_56:$ae_immls64), "ae_l32x2f24.i $ae_ls_v, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls64;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{6} = ae_immls64{2};
let Inst{7} = ae_immls64{3};
}



def AE_L32X2F24_I : AE_L32X2F24_I_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l32x2f24_i AR:$ars, timm:$ae_immls64))]>;

class AE_L32X2F24_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, imm0_56:$ae_immls64pos), "ae_l32x2f24.ip $ae_ls_v, $ars, $ae_immls64pos", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<3> ae_immls64pos;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64pos{0};
let Inst{5} = ae_immls64pos{1};
let Inst{6} = ae_immls64pos{2};
}



def AE_L32X2F24_IP : AE_L32X2F24_IP_X24<[]>;

class AE_L32X2F24_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars), "ae_l32x2f24.ric $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_L32X2F24_RIC : AE_L32X2F24_RIC_AE_FORMAT48<[]>;

class AE_L32X2F24_RIP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars), "ae_l32x2f24.rip $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{5} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_L32X2F24_RIP : AE_L32X2F24_RIP_X24<[]>;

class AE_L32X2F24_X_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, AR:$art), "ae_l32x2f24.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32X2F24_X : AE_L32X2F24_X_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l32x2f24_x AR:$ars, AR:$art))]>;

class AE_L32X2F24_XC_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l32x2f24.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32X2F24_XC : AE_L32X2F24_XC_X24<[]>;

class AE_L32X2F24_XP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l32x2f24.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L32X2F24_XP : AE_L32X2F24_XP_X24<[]>;

class AE_L64_I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v), (ins AR:$ars, imm64n_56:$ae_immls64), "ae_l64.i $ae_ls_v, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls64;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{6} = ae_immls64{2};
let Inst{7} = ae_immls64{3};
}



def AE_L64_I : AE_L64_I_X24<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l64_i AR:$ars, timm:$ae_immls64))]>;

class AE_L64_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, imm64n_56:$ae_immls64), "ae_l64.ip $ae_ls_v, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls64;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{6} = ae_immls64{2};
let Inst{7} = ae_immls64{3};
}



def AE_L64_IP : AE_L64_IP_X24<[]>;

class AE_L64_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v), (ins AR:$ars, AR:$art), "ae_l64.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;

let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L64_X : AE_L64_X_AE_FORMAT48<[(set AE_DR:$ae_ls_v, (int_xtensa_ae_l64_x AR:$ars, AR:$art))]>;

class AE_L64_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l64.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L64_XC : AE_L64_XC_AE_FORMAT48<[]>;

class AE_L64_XP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_v, AR:$ars_out), (ins AR:$ars, AR:$art), "ae_l64.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_L64_XP : AE_L64_XP_AE_FORMAT48<[]>;

class AE_LA16X4_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la16x4.ic $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA16X4_IC : AE_LA16X4_IC_X24<[]>;

class AE_LA16X4_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la16x4.ip $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA16X4_IP : AE_LA16X4_IP_X24<[]>;

class AE_LA16X4_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la16x4.ric $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA16X4_RIC : AE_LA16X4_RIC_AE_FORMAT48<[]>;

class AE_LA16X4_RIP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la16x4.rip $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA16X4_RIP : AE_LA16X4_RIP_X24<[]>;

class AE_LA16X4NEG_PC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_uu, AR:$ars_out), (ins AR:$ars), "ae_la16x4neg.pc $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA16X4NEG_PC : AE_LA16X4NEG_PC_AE_FORMAT48<[]>;

class AE_LA16X4POS_PC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_uu, AR:$ars_out), (ins AR:$ars), "ae_la16x4pos.pc $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA16X4POS_PC : AE_LA16X4POS_PC_AE_FORMAT48<[]>;

class AE_LA24_IC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la24.ic $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24_IC : AE_LA24_IC_AE_FORMAT48<[]>;

class AE_LA24_IP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la24.ip $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24_IP : AE_LA24_IP_AE_FORMAT48<[]>;

class AE_LA24_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la24.ric $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24_RIC : AE_LA24_RIC_AE_FORMAT48<[]>;

class AE_LA24_RIP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la24.rip $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24_RIP : AE_LA24_RIP_AE_FORMAT48<[]>;

class AE_LA24NEG_PC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_uu, AR:$ars_out), (ins AR:$ars), "ae_la24neg.pc $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24NEG_PC : AE_LA24NEG_PC_AE_FORMAT48<[]>;

class AE_LA24POS_PC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_uu, AR:$ars_out), (ins AR:$ars), "ae_la24pos.pc $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24POS_PC : AE_LA24POS_PC_AE_FORMAT48<[]>;

class AE_LA24X2_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la24x2.ic $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24X2_IC : AE_LA24X2_IC_X24<[]>;

class AE_LA24X2_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la24x2.ip $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24X2_IP : AE_LA24X2_IP_X24<[]>;

class AE_LA24X2_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la24x2.ric $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{29} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24X2_RIC : AE_LA24X2_RIC_AE_FORMAT48<[]>;

class AE_LA24X2_RIP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la24x2.rip $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24X2_RIP : AE_LA24X2_RIP_AE_FORMAT48<[]>;

class AE_LA24X2NEG_PC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_uu, AR:$ars_out), (ins AR:$ars), "ae_la24x2neg.pc $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24X2NEG_PC : AE_LA24X2NEG_PC_AE_FORMAT48<[]>;

class AE_LA24X2POS_PC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_uu, AR:$ars_out), (ins AR:$ars), "ae_la24x2pos.pc $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA24X2POS_PC : AE_LA24X2POS_PC_AE_FORMAT48<[]>;

class AE_LA32X2_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la32x2.ic $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA32X2_IC : AE_LA32X2_IC_X24<[]>;

class AE_LA32X2_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la32x2.ip $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA32X2_IP : AE_LA32X2_IP_X24<[]>;

class AE_LA32X2_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la32x2.ric $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA32X2_RIC : AE_LA32X2_RIC_AE_FORMAT48<[]>;

class AE_LA32X2_RIP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la32x2.rip $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA32X2_RIP : AE_LA32X2_RIP_AE_FORMAT48<[]>;

class AE_LA32X2F24_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la32x2f24.ic $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA32X2F24_IC : AE_LA32X2F24_IC_X24<[]>;

class AE_LA32X2F24_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la32x2f24.ip $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA32X2F24_IP : AE_LA32X2F24_IP_X24<[]>;

class AE_LA32X2F24_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la32x2f24.ric $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA32X2F24_RIC : AE_LA32X2F24_RIC_AE_FORMAT48<[]>;

class AE_LA32X2F24_RIP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$ae_ls_av, AE_VALIGN:$ae_ls_uu_out, AR:$ars_out), (ins AE_VALIGN:$ae_ls_uu, AR:$ars), "ae_la32x2f24.rip $ae_ls_av, $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_av;
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ae_ls_uu = $ae_ls_uu_out,@earlyclobber $ae_ls_uu_out, $ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_av{0};
let Inst{13} = ae_ls_av{1};
let Inst{14} = ae_ls_av{2};
let Inst{15} = ae_ls_av{3};
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA32X2F24_RIP : AE_LA32X2F24_RIP_AE_FORMAT48<[]>;

class AE_LA32X2NEG_PC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_uu, AR:$ars_out), (ins AR:$ars), "ae_la32x2neg.pc $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA32X2NEG_PC : AE_LA32X2NEG_PC_AE_FORMAT48<[]>;

class AE_LA32X2POS_PC_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_VALIGN:$ae_ls_uu, AR:$ars_out), (ins AR:$ars), "ae_la32x2pos.pc $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_uu;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";
let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{12} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA32X2POS_PC : AE_LA32X2POS_PC_X24<[]>;

class AE_LA64_PP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_VALIGN:$ae_ls_uu), (ins AR:$ars), "ae_la64.pp $ae_ls_uu, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_uu;
bits<4> ars;

let mayLoad = 1;

//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{13} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = ae_ls_uu{0};
let Inst{5} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_LA64_PP : AE_LA64_PP_X24<[(set AE_VALIGN:$ae_ls_uu, (int_xtensa_ae_la64_pp AR:$ars))]>;

class AE_LALIGN64_I_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_uu), (ins AR:$ars, imm64n_56:$ae_immls64), "ae_lalign64.i $ae_ls_uu, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_uu;
bits<4> ars;
bits<4> ae_immls64;

let mayLoad = 1;

//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{30} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{6} = ae_ls_uu{0};
let Inst{7} = ae_ls_uu{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{28} = ae_immls64{2};
let Inst{29} = ae_immls64{3};
}



def AE_LALIGN64_I : AE_LALIGN64_I_AE_FORMAT48<[(set AE_VALIGN:$ae_ls_uu, (int_xtensa_ae_lalign64_i AR:$ars, timm:$ae_immls64))]>;

class AE_LB_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AR:$art), "ae_lb $arr, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{16} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_LB : AE_LB_X24<[(set AR:$arr, (int_xtensa_ae_lb AR:$art))]>;

class AE_LBI_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins imm1_16:$ae_ohba), "ae_lbi $arr, $ae_ohba", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_ohba;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{16} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{4} = ae_ohba{0};
let Inst{5} = ae_ohba{1};
let Inst{6} = ae_ohba{2};
let Inst{7} = ae_ohba{3};
}



def AE_LBI : AE_LBI_X24<[(set AR:$arr, (int_xtensa_ae_lbi timm:$ae_ohba))]>;

class AE_LBK_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AR:$ars, AR:$art), "ae_lbk $arr, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ars;
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_LBK : AE_LBK_X24<[(set AR:$arr, (int_xtensa_ae_lbk AR:$ars, AR:$art))]>;

class AE_LBKI_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AR:$ars, imm1_16:$ae_ohba), "ae_lbki $arr, $ars, $ae_ohba", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ars;
bits<4> ae_ohba;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_ohba{0};
let Inst{5} = ae_ohba{1};
let Inst{6} = ae_ohba{2};
let Inst{7} = ae_ohba{3};
}



def AE_LBKI : AE_LBKI_X24<[(set AR:$arr, (int_xtensa_ae_lbki AR:$ars, timm:$ae_ohba))]>;

class AE_LBS_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AR:$art), "ae_lbs $arr, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{16} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_LBS : AE_LBS_X24<[(set AR:$arr, (int_xtensa_ae_lbs AR:$art))]>;

class AE_LBSI_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins imm1_16:$ae_ohba), "ae_lbsi $arr, $ae_ohba", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_ohba;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{16} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{4} = ae_ohba{0};
let Inst{5} = ae_ohba{1};
let Inst{6} = ae_ohba{2};
let Inst{7} = ae_ohba{3};
}



def AE_LBSI : AE_LBSI_X24<[(set AR:$arr, (int_xtensa_ae_lbsi timm:$ae_ohba))]>;

class AE_LE16_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs BR4:$br4), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_le16 $br4, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> br4;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{62} = 1;
//operands
let Inst{22} = br4{0};
let Inst{23} = br4{1};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_LE16 : AE_LE16_AE_FORMAT1<[(set BR4:$br4, (int_xtensa_ae_le16 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_LE32_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs BR2:$br2), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_le32 $br2, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<3> br2;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{20} = 1;
let Inst{57} = 1;
let Inst{62} = 1;
//operands
let Inst{21} = br2{0};
let Inst{22} = br2{1};
let Inst{23} = br2{2};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_LE32 : AE_LE32_AE_FORMAT1<[(set BR2:$br2, (int_xtensa_ae_le32 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_LE64_X24<list<dag> pattern>
    : XtensaInst24<(outs BR:$br), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_le64 $br, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> br;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = br{0};
let Inst{13} = br{1};
let Inst{14} = br{2};
let Inst{15} = br{3};
let Inst{8} = ae_cmpp_v0{0};
let Inst{9} = ae_cmpp_v0{1};
let Inst{10} = ae_cmpp_v0{2};
let Inst{11} = ae_cmpp_v0{3};
let Inst{4} = ae_cmpp_v1{0};
let Inst{5} = ae_cmpp_v1{1};
let Inst{6} = ae_cmpp_v1{2};
let Inst{7} = ae_cmpp_v1{3};
}



def AE_LE64 : AE_LE64_X24<[(set BR:$br, (int_xtensa_ae_le64 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_LT16_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs BR4:$br4), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_lt16 $br4, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> br4;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
//operands
let Inst{22} = br4{0};
let Inst{23} = br4{1};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_LT16 : AE_LT16_AE_FORMAT1<[(set BR4:$br4, (int_xtensa_ae_lt16 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_LT32_X24<list<dag> pattern>
    : XtensaInst24<(outs BR2:$br2), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_lt32 $br2, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<3> br2;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{12} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
//operands
let Inst{13} = br2{0};
let Inst{14} = br2{1};
let Inst{15} = br2{2};
let Inst{8} = ae_cmpp_v0{0};
let Inst{9} = ae_cmpp_v0{1};
let Inst{10} = ae_cmpp_v0{2};
let Inst{11} = ae_cmpp_v0{3};
let Inst{4} = ae_cmpp_v1{0};
let Inst{5} = ae_cmpp_v1{1};
let Inst{6} = ae_cmpp_v1{2};
let Inst{7} = ae_cmpp_v1{3};
}



def AE_LT32 : AE_LT32_X24<[(set BR2:$br2, (int_xtensa_ae_lt32 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_LT64_X24<list<dag> pattern>
    : XtensaInst24<(outs BR:$br), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_lt64 $br, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> br;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = br{0};
let Inst{13} = br{1};
let Inst{14} = br{2};
let Inst{15} = br{3};
let Inst{8} = ae_cmpp_v0{0};
let Inst{9} = ae_cmpp_v0{1};
let Inst{10} = ae_cmpp_v0{2};
let Inst{11} = ae_cmpp_v0{3};
let Inst{4} = ae_cmpp_v1{0};
let Inst{5} = ae_cmpp_v1{1};
let Inst{6} = ae_cmpp_v1{2};
let Inst{7} = ae_cmpp_v1{3};
}



def AE_LT64 : AE_LT64_X24<[(set BR:$br, (int_xtensa_ae_lt64 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_MAX32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_cmpp_v), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_max32 $ae_cmpp_v, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmpp_v;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_cmpp_v{0};
let Inst{13} = ae_cmpp_v{1};
let Inst{14} = ae_cmpp_v{2};
let Inst{15} = ae_cmpp_v{3};
let Inst{8} = ae_cmpp_v0{0};
let Inst{9} = ae_cmpp_v0{1};
let Inst{10} = ae_cmpp_v0{2};
let Inst{11} = ae_cmpp_v0{3};
let Inst{4} = ae_cmpp_v1{0};
let Inst{5} = ae_cmpp_v1{1};
let Inst{6} = ae_cmpp_v1{2};
let Inst{7} = ae_cmpp_v1{3};
}



def AE_MAX32 : AE_MAX32_X24<[(set AE_DR:$ae_cmpp_v, (int_xtensa_ae_max32 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_MAX64_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_cmpp_v), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_max64 $ae_cmpp_v, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmpp_v;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_cmpp_v{0};
let Inst{21} = ae_cmpp_v{1};
let Inst{22} = ae_cmpp_v{2};
let Inst{23} = ae_cmpp_v{3};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_MAX64 : AE_MAX64_AE_FORMAT1<[(set AE_DR:$ae_cmpp_v, (int_xtensa_ae_max64 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_MAXABS32S_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_cmpp_v), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_maxabs32s $ae_cmpp_v, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmpp_v;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_cmpp_v{0};
let Inst{21} = ae_cmpp_v{1};
let Inst{22} = ae_cmpp_v{2};
let Inst{23} = ae_cmpp_v{3};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_MAXABS32S : AE_MAXABS32S_AE_FORMAT1<[(set AE_DR:$ae_cmpp_v, (int_xtensa_ae_maxabs32s AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_MAXABS64S_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_cmpp_v), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_maxabs64s $ae_cmpp_v, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmpp_v;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_cmpp_v{0};
let Inst{21} = ae_cmpp_v{1};
let Inst{22} = ae_cmpp_v{2};
let Inst{23} = ae_cmpp_v{3};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_MAXABS64S : AE_MAXABS64S_AE_FORMAT1<[(set AE_DR:$ae_cmpp_v, (int_xtensa_ae_maxabs64s AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_MIN32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_cmpp_v), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_min32 $ae_cmpp_v, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmpp_v;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_cmpp_v{0};
let Inst{13} = ae_cmpp_v{1};
let Inst{14} = ae_cmpp_v{2};
let Inst{15} = ae_cmpp_v{3};
let Inst{8} = ae_cmpp_v0{0};
let Inst{9} = ae_cmpp_v0{1};
let Inst{10} = ae_cmpp_v0{2};
let Inst{11} = ae_cmpp_v0{3};
let Inst{4} = ae_cmpp_v1{0};
let Inst{5} = ae_cmpp_v1{1};
let Inst{6} = ae_cmpp_v1{2};
let Inst{7} = ae_cmpp_v1{3};
}



def AE_MIN32 : AE_MIN32_X24<[(set AE_DR:$ae_cmpp_v, (int_xtensa_ae_min32 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_MIN64_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_cmpp_v), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_min64 $ae_cmpp_v, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmpp_v;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_cmpp_v{0};
let Inst{21} = ae_cmpp_v{1};
let Inst{22} = ae_cmpp_v{2};
let Inst{23} = ae_cmpp_v{3};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_MIN64 : AE_MIN64_AE_FORMAT1<[(set AE_DR:$ae_cmpp_v, (int_xtensa_ae_min64 AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_MINABS32S_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_cmpp_v), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_minabs32s $ae_cmpp_v, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmpp_v;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_cmpp_v{0};
let Inst{21} = ae_cmpp_v{1};
let Inst{22} = ae_cmpp_v{2};
let Inst{23} = ae_cmpp_v{3};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_MINABS32S : AE_MINABS32S_AE_FORMAT1<[(set AE_DR:$ae_cmpp_v, (int_xtensa_ae_minabs32s AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_MINABS64S_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_cmpp_v), (ins AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1), "ae_minabs64s $ae_cmpp_v, $ae_cmpp_v0, $ae_cmpp_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmpp_v;
bits<4> ae_cmpp_v0;
bits<4> ae_cmpp_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_cmpp_v{0};
let Inst{21} = ae_cmpp_v{1};
let Inst{22} = ae_cmpp_v{2};
let Inst{23} = ae_cmpp_v{3};
let Inst{36} = ae_cmpp_v0{0};
let Inst{37} = ae_cmpp_v0{1};
let Inst{38} = ae_cmpp_v0{2};
let Inst{39} = ae_cmpp_v0{3};
let Inst{28} = ae_cmpp_v1{0};
let Inst{29} = ae_cmpp_v1{1};
let Inst{30} = ae_cmpp_v1{2};
let Inst{31} = ae_cmpp_v1{3};
}



def AE_MINABS64S : AE_MINABS64S_AE_FORMAT1<[(set AE_DR:$ae_cmpp_v, (int_xtensa_ae_minabs64s AE_DR:$ae_cmpp_v0, AE_DR:$ae_cmpp_v1))]>;

class AE_MOV_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_to_dr_v), (ins AE_DR:$ae_to_dr_v0), "ae_mov $ae_to_dr_v, $ae_to_dr_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_to_dr_v;
bits<4> ae_to_dr_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_to_dr_v{0};
let Inst{13} = ae_to_dr_v{1};
let Inst{14} = ae_to_dr_v{2};
let Inst{15} = ae_to_dr_v{3};
let Inst{8} = ae_to_dr_v0{0};
let Inst{9} = ae_to_dr_v0{1};
let Inst{10} = ae_to_dr_v0{2};
let Inst{11} = ae_to_dr_v0{3};
}



def AE_MOV : AE_MOV_X24<[(set AE_DR:$ae_to_dr_v, (int_xtensa_ae_mov AE_DR:$ae_to_dr_v0))]>;

class AE_MOVAD16_0_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_movad16.0 $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ae_dr_to_ar_v0{0};
let Inst{9} = ae_dr_to_ar_v0{1};
let Inst{10} = ae_dr_to_ar_v0{2};
let Inst{11} = ae_dr_to_ar_v0{3};
}



def AE_MOVAD16_0 : AE_MOVAD16_0_X24<[(set AR:$arr, (int_xtensa_ae_movad16_0 AE_DR:$ae_dr_to_ar_v0))]>;

class AE_MOVAD16_1_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_movad16.1 $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{11} = 1;
let Inst{31} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{4} = ae_dr_to_ar_v0{0};
let Inst{5} = ae_dr_to_ar_v0{1};
let Inst{6} = ae_dr_to_ar_v0{2};
let Inst{7} = ae_dr_to_ar_v0{3};
}



def AE_MOVAD16_1 : AE_MOVAD16_1_AE_FORMAT48<[(set AR:$arr, (int_xtensa_ae_movad16_1 AE_DR:$ae_dr_to_ar_v0))]>;

class AE_MOVAD16_2_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_movad16.2 $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ae_dr_to_ar_v0{0};
let Inst{9} = ae_dr_to_ar_v0{1};
let Inst{10} = ae_dr_to_ar_v0{2};
let Inst{11} = ae_dr_to_ar_v0{3};
}



def AE_MOVAD16_2 : AE_MOVAD16_2_X24<[(set AR:$arr, (int_xtensa_ae_movad16_2 AE_DR:$ae_dr_to_ar_v0))]>;

class AE_MOVAD16_3_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_movad16.3 $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{5} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ae_dr_to_ar_v0{0};
let Inst{9} = ae_dr_to_ar_v0{1};
let Inst{10} = ae_dr_to_ar_v0{2};
let Inst{11} = ae_dr_to_ar_v0{3};
}



def AE_MOVAD16_3 : AE_MOVAD16_3_X24<[(set AR:$arr, (int_xtensa_ae_movad16_3 AE_DR:$ae_dr_to_ar_v0))]>;

class AE_MOVAD32_H_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_movad32.h $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ae_dr_to_ar_v0{0};
let Inst{9} = ae_dr_to_ar_v0{1};
let Inst{10} = ae_dr_to_ar_v0{2};
let Inst{11} = ae_dr_to_ar_v0{3};
}



def AE_MOVAD32_H : AE_MOVAD32_H_X24<[(set AR:$arr, (int_xtensa_ae_movad32_h AE_DR:$ae_dr_to_ar_v0))]>;

class AE_MOVAD32_L_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_movad32.l $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ae_dr_to_ar_v0{0};
let Inst{9} = ae_dr_to_ar_v0{1};
let Inst{10} = ae_dr_to_ar_v0{2};
let Inst{11} = ae_dr_to_ar_v0{3};
}



def AE_MOVAD32_L : AE_MOVAD32_L_X24<[(set AR:$arr, (int_xtensa_ae_movad32_l AE_DR:$ae_dr_to_ar_v0))]>;

class AE_MOVALIGN_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_VALIGN:$ae_uu_uu), (ins AE_VALIGN:$ae_uu_v), "ae_movalign $ae_uu_uu, $ae_uu_v", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_uu_uu;
bits<2> ae_uu_v;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{6} = ae_uu_uu{0};
let Inst{7} = ae_uu_uu{1};
let Inst{4} = ae_uu_v{0};
let Inst{5} = ae_uu_v{1};
}



def AE_MOVALIGN : AE_MOVALIGN_X24<[(set AE_VALIGN:$ae_uu_uu, (int_xtensa_ae_movalign AE_VALIGN:$ae_uu_v))]>;

class AE_MOVDA16_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ar_to_dr_v), (ins AR:$ars), "ae_movda16 $ae_ar_to_dr_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ar_to_dr_v;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{5} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ar_to_dr_v{0};
let Inst{13} = ae_ar_to_dr_v{1};
let Inst{14} = ae_ar_to_dr_v{2};
let Inst{15} = ae_ar_to_dr_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_MOVDA16 : AE_MOVDA16_X24<[(set AE_DR:$ae_ar_to_dr_v, (int_xtensa_ae_movda16 AR:$ars))]>;

class AE_MOVDA16X2_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ar_to_dr_v), (ins AR:$ars, AR:$art), "ae_movda16x2 $ae_ar_to_dr_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ar_to_dr_v;
bits<4> ars;
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
//operands
let Inst{12} = ae_ar_to_dr_v{0};
let Inst{13} = ae_ar_to_dr_v{1};
let Inst{14} = ae_ar_to_dr_v{2};
let Inst{15} = ae_ar_to_dr_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_MOVDA16X2 : AE_MOVDA16X2_X24<[(set AE_DR:$ae_ar_to_dr_v, (int_xtensa_ae_movda16x2 AR:$ars, AR:$art))]>;

class AE_MOVDA32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ar_to_dr_v), (ins AR:$ars), "ae_movda32 $ae_ar_to_dr_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ar_to_dr_v;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ar_to_dr_v{0};
let Inst{13} = ae_ar_to_dr_v{1};
let Inst{14} = ae_ar_to_dr_v{2};
let Inst{15} = ae_ar_to_dr_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_MOVDA32 : AE_MOVDA32_X24<[(set AE_DR:$ae_ar_to_dr_v, (int_xtensa_ae_movda32 AR:$ars))]>;

class AE_MOVDA32X2_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ar_to_dr_v), (ins AR:$ars, AR:$art), "ae_movda32x2 $ae_ar_to_dr_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ar_to_dr_v;
bits<4> ars;
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
//operands
let Inst{12} = ae_ar_to_dr_v{0};
let Inst{13} = ae_ar_to_dr_v{1};
let Inst{14} = ae_ar_to_dr_v{2};
let Inst{15} = ae_ar_to_dr_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_MOVDA32X2 : AE_MOVDA32X2_X24<[(set AE_DR:$ae_ar_to_dr_v, (int_xtensa_ae_movda32x2 AR:$ars, AR:$art))]>;

class AE_MOVF16X4_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_cmov_v_out), (ins AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR4:$bt4), "ae_movf16x4 $ae_cmov_v, $ae_cmov_v0, $bt4", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmov_v;
bits<4> ae_cmov_v0;
bits<2> bt4;
let Constraints = "$ae_cmov_v = $ae_cmov_v_out,@earlyclobber $ae_cmov_v_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_cmov_v{0};
let Inst{21} = ae_cmov_v{1};
let Inst{22} = ae_cmov_v{2};
let Inst{23} = ae_cmov_v{3};
let Inst{36} = ae_cmov_v0{0};
let Inst{37} = ae_cmov_v0{1};
let Inst{38} = ae_cmov_v0{2};
let Inst{39} = ae_cmov_v0{3};
let Inst{30} = bt4{0};
let Inst{31} = bt4{1};
}



def AE_MOVF16X4 : AE_MOVF16X4_AE_FORMAT1<[(set AE_DR:$ae_cmov_v_out, (int_xtensa_ae_movf16x4 AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR4:$bt4))]>;

class AE_MOVF32X2_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_cmov_v_out), (ins AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR2:$bt2), "ae_movf32x2 $ae_cmov_v, $ae_cmov_v0, $bt2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmov_v;
bits<4> ae_cmov_v0;
bits<3> bt2;
let Constraints = "$ae_cmov_v = $ae_cmov_v_out,@earlyclobber $ae_cmov_v_out";


//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_cmov_v{0};
let Inst{13} = ae_cmov_v{1};
let Inst{14} = ae_cmov_v{2};
let Inst{15} = ae_cmov_v{3};
let Inst{8} = ae_cmov_v0{0};
let Inst{9} = ae_cmov_v0{1};
let Inst{10} = ae_cmov_v0{2};
let Inst{11} = ae_cmov_v0{3};
let Inst{5} = bt2{0};
let Inst{6} = bt2{1};
let Inst{7} = bt2{2};
}



def AE_MOVF32X2 : AE_MOVF32X2_X24<[(set AE_DR:$ae_cmov_v_out, (int_xtensa_ae_movf32x2 AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR2:$bt2))]>;

class AE_MOVF64_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_cmov_v_out), (ins AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR:$bt), "ae_movf64 $ae_cmov_v, $ae_cmov_v0, $bt", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmov_v;
bits<4> ae_cmov_v0;
bits<4> bt;
let Constraints = "$ae_cmov_v = $ae_cmov_v_out,@earlyclobber $ae_cmov_v_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{62} = 1;
//operands
let Inst{20} = ae_cmov_v{0};
let Inst{21} = ae_cmov_v{1};
let Inst{22} = ae_cmov_v{2};
let Inst{23} = ae_cmov_v{3};
let Inst{36} = ae_cmov_v0{0};
let Inst{37} = ae_cmov_v0{1};
let Inst{38} = ae_cmov_v0{2};
let Inst{39} = ae_cmov_v0{3};
let Inst{28} = bt{0};
let Inst{29} = bt{1};
let Inst{30} = bt{2};
let Inst{31} = bt{3};
}



def AE_MOVF64 : AE_MOVF64_AE_FORMAT1<[(set AE_DR:$ae_cmov_v_out, (int_xtensa_ae_movf64 AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR:$bt))]>;

class AE_MOVI_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_ar_to_dr_v), (ins imm16n_47:$movi_imm), "ae_movi $ae_ar_to_dr_v, $movi_imm", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ar_to_dr_v;
bits<6> movi_imm;



//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ar_to_dr_v{0};
let Inst{13} = ae_ar_to_dr_v{1};
let Inst{14} = ae_ar_to_dr_v{2};
let Inst{15} = ae_ar_to_dr_v{3};
let Inst{4} = movi_imm{0};
let Inst{5} = movi_imm{1};
let Inst{8} = movi_imm{2};
let Inst{9} = movi_imm{3};
let Inst{10} = movi_imm{4};
let Inst{11} = movi_imm{5};
}



def AE_MOVI : AE_MOVI_X24<[(set AE_DR:$ae_ar_to_dr_v, (int_xtensa_ae_movi timm:$movi_imm))]>;

class AE_MOVT16X4_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_cmov_v_out), (ins AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR4:$bt4), "ae_movt16x4 $ae_cmov_v, $ae_cmov_v0, $bt4", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmov_v;
bits<4> ae_cmov_v0;
bits<2> bt4;
let Constraints = "$ae_cmov_v = $ae_cmov_v_out,@earlyclobber $ae_cmov_v_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{28} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_cmov_v{0};
let Inst{21} = ae_cmov_v{1};
let Inst{22} = ae_cmov_v{2};
let Inst{23} = ae_cmov_v{3};
let Inst{36} = ae_cmov_v0{0};
let Inst{37} = ae_cmov_v0{1};
let Inst{38} = ae_cmov_v0{2};
let Inst{39} = ae_cmov_v0{3};
let Inst{30} = bt4{0};
let Inst{31} = bt4{1};
}



def AE_MOVT16X4 : AE_MOVT16X4_AE_FORMAT1<[(set AE_DR:$ae_cmov_v_out, (int_xtensa_ae_movt16x4 AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR4:$bt4))]>;

class AE_MOVT32X2_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_cmov_v_out), (ins AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR2:$bt2), "ae_movt32x2 $ae_cmov_v, $ae_cmov_v0, $bt2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmov_v;
bits<4> ae_cmov_v0;
bits<3> bt2;
let Constraints = "$ae_cmov_v = $ae_cmov_v_out,@earlyclobber $ae_cmov_v_out";


//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_cmov_v{0};
let Inst{13} = ae_cmov_v{1};
let Inst{14} = ae_cmov_v{2};
let Inst{15} = ae_cmov_v{3};
let Inst{8} = ae_cmov_v0{0};
let Inst{9} = ae_cmov_v0{1};
let Inst{10} = ae_cmov_v0{2};
let Inst{11} = ae_cmov_v0{3};
let Inst{5} = bt2{0};
let Inst{6} = bt2{1};
let Inst{7} = bt2{2};
}



def AE_MOVT32X2 : AE_MOVT32X2_X24<[(set AE_DR:$ae_cmov_v_out, (int_xtensa_ae_movt32x2 AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR2:$bt2))]>;

class AE_MOVT64_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_cmov_v_out), (ins AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR:$bt), "ae_movt64 $ae_cmov_v, $ae_cmov_v0, $bt", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_cmov_v;
bits<4> ae_cmov_v0;
bits<4> bt;
let Constraints = "$ae_cmov_v = $ae_cmov_v_out,@earlyclobber $ae_cmov_v_out";


//format

//opcode
let Inst{2} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_cmov_v{0};
let Inst{13} = ae_cmov_v{1};
let Inst{14} = ae_cmov_v{2};
let Inst{15} = ae_cmov_v{3};
let Inst{8} = ae_cmov_v0{0};
let Inst{9} = ae_cmov_v0{1};
let Inst{10} = ae_cmov_v0{2};
let Inst{11} = ae_cmov_v0{3};
let Inst{4} = bt{0};
let Inst{5} = bt{1};
let Inst{6} = bt{2};
let Inst{7} = bt{3};
}



def AE_MOVT64 : AE_MOVT64_X24<[(set AE_DR:$ae_cmov_v_out, (int_xtensa_ae_movt64 AE_DR:$ae_cmov_v, AE_DR:$ae_cmov_v0, BR:$bt))]>;

class AE_MUL16X4_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q1, AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d0), "ae_mul16x4 $ae_mul_q1, $ae_mul_q0, $ae_mul_d1, $ae_mul_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q1;
bits<4> ae_mul_q0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d0;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
}



def AE_MUL16X4 : AE_MUL16X4_AE_FORMAT2<[]>;

class AE_MUL32_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{42} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32_HH : AE_MUL32_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32_hh AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32_LH : AE_MUL32_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32_lh AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32_LL : AE_MUL32_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32_ll AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mul32.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MUL32_LL_S2 : AE_MUL32_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mul32_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MUL32U_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32u.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32U_LL : AE_MUL32U_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32u_ll AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32X16_H0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32x16.h0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32X16_H0 : AE_MUL32X16_H0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32x16_h0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32X16_H0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mul32x16.h0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MUL32X16_H0_S2 : AE_MUL32X16_H0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mul32x16_h0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MUL32X16_H1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32x16.h1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32X16_H1 : AE_MUL32X16_H1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32x16_h1 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32X16_H1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mul32x16.h1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MUL32X16_H1_S2 : AE_MUL32X16_H1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mul32x16_h1_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MUL32X16_H2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32x16.h2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32X16_H2 : AE_MUL32X16_H2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32x16_h2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32X16_H2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mul32x16.h2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MUL32X16_H2_S2 : AE_MUL32X16_H2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mul32x16_h2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MUL32X16_H3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32x16.h3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32X16_H3 : AE_MUL32X16_H3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32x16_h3 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32X16_H3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mul32x16.h3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MUL32X16_H3_S2 : AE_MUL32X16_H3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mul32x16_h3_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MUL32X16_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32x16.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32X16_L0 : AE_MUL32X16_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32x16_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32X16_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mul32x16.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MUL32X16_L0_S2 : AE_MUL32X16_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mul32x16_l0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MUL32X16_L1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32x16.l1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32X16_L1 : AE_MUL32X16_L1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32x16_l1 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32X16_L1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mul32x16.l1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MUL32X16_L1_S2 : AE_MUL32X16_L1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mul32x16_l1_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MUL32X16_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32x16.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32X16_L2 : AE_MUL32X16_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32x16_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32X16_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mul32x16.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MUL32X16_L2_S2 : AE_MUL32X16_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mul32x16_l2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MUL32X16_L3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mul32x16.l3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MUL32X16_L3 : AE_MUL32X16_L3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mul32x16_l3 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MUL32X16_L3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mul32x16.l3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MUL32X16_L3_S2 : AE_MUL32X16_L3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mul32x16_l3_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULA16X4_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q1_out, AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q1, AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d0), "ae_mula16x4 $ae_mul_q1, $ae_mul_q0, $ae_mul_d1, $ae_mul_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q1;
bits<4> ae_mul_q0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d0;
let Constraints = "$ae_mul_q1 = $ae_mul_q1_out,@earlyclobber $ae_mul_q1_out, $ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
}



def AE_MULA16X4 : AE_MULA16X4_AE_FORMAT2<[]>;

class AE_MULA32_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32_HH : AE_MULA32_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32_hh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32_LH : AE_MULA32_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32_lh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32_LL : AE_MULA32_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32_ll AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mula32.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULA32_LL_S2 : AE_MULA32_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mula32_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULA32U_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32u.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32U_LL : AE_MULA32U_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32u_ll AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32X16_H0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32x16.h0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32X16_H0 : AE_MULA32X16_H0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32x16_h0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32X16_H0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mula32x16.h0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULA32X16_H0_S2 : AE_MULA32X16_H0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mula32x16_h0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULA32X16_H1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32x16.h1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32X16_H1 : AE_MULA32X16_H1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32x16_h1 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32X16_H1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mula32x16.h1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULA32X16_H1_S2 : AE_MULA32X16_H1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mula32x16_h1_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULA32X16_H2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32x16.h2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32X16_H2 : AE_MULA32X16_H2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32x16_h2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32X16_H2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mula32x16.h2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULA32X16_H2_S2 : AE_MULA32X16_H2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mula32x16_h2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULA32X16_H3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32x16.h3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32X16_H3 : AE_MULA32X16_H3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32x16_h3 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32X16_H3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mula32x16.h3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULA32X16_H3_S2 : AE_MULA32X16_H3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mula32x16_h3_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULA32X16_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32x16.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32X16_L0 : AE_MULA32X16_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32x16_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32X16_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mula32x16.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULA32X16_L0_S2 : AE_MULA32X16_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mula32x16_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULA32X16_L1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32x16.l1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32X16_L1 : AE_MULA32X16_L1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32x16_l1 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32X16_L1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mula32x16.l1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULA32X16_L1_S2 : AE_MULA32X16_L1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mula32x16_l1_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULA32X16_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32x16.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32X16_L2 : AE_MULA32X16_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32x16_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32X16_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mula32x16.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULA32X16_L2_S2 : AE_MULA32X16_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mula32x16_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULA32X16_L3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mula32x16.l3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULA32X16_L3 : AE_MULA32X16_L3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mula32x16_l3 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULA32X16_L3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mula32x16.l3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULA32X16_L3_S2 : AE_MULA32X16_L3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mula32x16_l3_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaad24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAAD24_HH_LL : AE_MULAAD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaad24_hh_ll AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAAD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaad24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAD24_HH_LL_S2 : AE_MULAAD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaad24_hh_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaad24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAAD24_HL_LH : AE_MULAAD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaad24_hl_lh AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAAD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaad24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAD24_HL_LH_S2 : AE_MULAAD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaad24_hl_lh_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAD32X16_H0_L1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaad32x16.h0.l1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAD32X16_H0_L1 : AE_MULAAD32X16_H0_L1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaad32x16_h0_l1 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAD32X16_H0_L1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaad32x16.h0.l1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAD32X16_H0_L1_S2 : AE_MULAAD32X16_H0_L1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaad32x16_h0_l1_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaad32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAD32X16_H1_L0 : AE_MULAAD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaad32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaad32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAD32X16_H1_L0_S2 : AE_MULAAD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaad32x16_h1_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAD32X16_H2_L3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaad32x16.h2.l3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAD32X16_H2_L3 : AE_MULAAD32X16_H2_L3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaad32x16_h2_l3 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAD32X16_H2_L3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaad32x16.h2.l3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAD32X16_H2_L3_S2 : AE_MULAAD32X16_H2_L3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaad32x16_h2_l3_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaad32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAD32X16_H3_L2 : AE_MULAAD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaad32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaad32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAD32X16_H3_L2_S2 : AE_MULAAD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaad32x16_h3_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAFD16SS_11_00_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaafd16ss.11_00 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAFD16SS_11_00 : AE_MULAAFD16SS_11_00_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaafd16ss_11_00 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAFD16SS_11_00_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaafd16ss.11_00_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAFD16SS_11_00_S2 : AE_MULAAFD16SS_11_00_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaafd16ss_11_00_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAFD16SS_13_02_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaafd16ss.13_02 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAFD16SS_13_02 : AE_MULAAFD16SS_13_02_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaafd16ss_13_02 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAFD16SS_13_02_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaafd16ss.13_02_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAFD16SS_13_02_S2 : AE_MULAAFD16SS_13_02_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaafd16ss_13_02_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAFD16SS_33_22_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaafd16ss.33_22 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAFD16SS_33_22 : AE_MULAAFD16SS_33_22_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaafd16ss_33_22 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAFD16SS_33_22_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaafd16ss.33_22_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAFD16SS_33_22_S2 : AE_MULAAFD16SS_33_22_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaafd16ss_33_22_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAFD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaafd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{50} = 1;
let Inst{53} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAAFD24_HH_LL : AE_MULAAFD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaafd24_hh_ll AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAAFD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaafd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAFD24_HH_LL_S2 : AE_MULAAFD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaafd24_hh_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAFD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaafd24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{53} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAAFD24_HL_LH : AE_MULAAFD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaafd24_hl_lh AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAAFD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaafd24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAFD24_HL_LH_S2 : AE_MULAAFD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaafd24_hl_lh_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAFD32X16_H0_L1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaafd32x16.h0.l1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAFD32X16_H0_L1 : AE_MULAAFD32X16_H0_L1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaafd32x16_h0_l1 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAFD32X16_H0_L1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaafd32x16.h0.l1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAFD32X16_H0_L1_S2 : AE_MULAAFD32X16_H0_L1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaafd32x16_h0_l1_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAFD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaafd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAFD32X16_H1_L0 : AE_MULAAFD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaafd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAFD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaafd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAFD32X16_H1_L0_S2 : AE_MULAAFD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaafd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAFD32X16_H2_L3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaafd32x16.h2.l3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAFD32X16_H2_L3 : AE_MULAAFD32X16_H2_L3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaafd32x16_h2_l3 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAFD32X16_H2_L3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaafd32x16.h2.l3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAFD32X16_H2_L3_S2 : AE_MULAAFD32X16_H2_L3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaafd32x16_h2_l3_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAAFD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaafd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAAFD32X16_H3_L2 : AE_MULAAFD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaafd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAAFD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaafd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAAFD32X16_H3_L2_S2 : AE_MULAAFD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaafd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAC24_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulac24 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{53} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAC24 : AE_MULAC24_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulac24 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAC32X16_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x4_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x4_q0, AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1), "ae_mulac32x16.h $opnd_ae_sem_mul_x4_q0, $opnd_ae_sem_mul_x4_d0, $opnd_ae_sem_mul_x4_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x4_q0;
bits<4> opnd_ae_sem_mul_x4_d0;
bits<4> opnd_ae_sem_mul_x4_d1;
let Constraints = "$opnd_ae_sem_mul_x4_q0 = $opnd_ae_sem_mul_x4_q0_out,@earlyclobber $opnd_ae_sem_mul_x4_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x4_q0{0};
let Inst{17} = opnd_ae_sem_mul_x4_q0{1};
let Inst{18} = opnd_ae_sem_mul_x4_q0{2};
let Inst{19} = opnd_ae_sem_mul_x4_q0{3};
let Inst{24} = opnd_ae_sem_mul_x4_d0{0};
let Inst{25} = opnd_ae_sem_mul_x4_d0{1};
let Inst{26} = opnd_ae_sem_mul_x4_d0{2};
let Inst{27} = opnd_ae_sem_mul_x4_d0{3};
let Inst{20} = opnd_ae_sem_mul_x4_d1{0};
let Inst{21} = opnd_ae_sem_mul_x4_d1{1};
let Inst{22} = opnd_ae_sem_mul_x4_d1{2};
let Inst{23} = opnd_ae_sem_mul_x4_d1{3};
}



def AE_MULAC32X16_H : AE_MULAC32X16_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x4_q0_out, (int_xtensa_ae_mulac32x16_h AE_DR:$opnd_ae_sem_mul_x4_q0, AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1))]>;

class AE_MULAC32X16_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x4_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x4_q0, AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1), "ae_mulac32x16.l $opnd_ae_sem_mul_x4_q0, $opnd_ae_sem_mul_x4_d0, $opnd_ae_sem_mul_x4_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x4_q0;
bits<4> opnd_ae_sem_mul_x4_d0;
bits<4> opnd_ae_sem_mul_x4_d1;
let Constraints = "$opnd_ae_sem_mul_x4_q0 = $opnd_ae_sem_mul_x4_q0_out,@earlyclobber $opnd_ae_sem_mul_x4_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x4_q0{0};
let Inst{17} = opnd_ae_sem_mul_x4_q0{1};
let Inst{18} = opnd_ae_sem_mul_x4_q0{2};
let Inst{19} = opnd_ae_sem_mul_x4_q0{3};
let Inst{24} = opnd_ae_sem_mul_x4_d0{0};
let Inst{25} = opnd_ae_sem_mul_x4_d0{1};
let Inst{26} = opnd_ae_sem_mul_x4_d0{2};
let Inst{27} = opnd_ae_sem_mul_x4_d0{3};
let Inst{20} = opnd_ae_sem_mul_x4_d1{0};
let Inst{21} = opnd_ae_sem_mul_x4_d1{1};
let Inst{22} = opnd_ae_sem_mul_x4_d1{2};
let Inst{23} = opnd_ae_sem_mul_x4_d1{3};
}



def AE_MULAC32X16_L : AE_MULAC32X16_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x4_q0_out, (int_xtensa_ae_mulac32x16_l AE_DR:$opnd_ae_sem_mul_x4_q0, AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1))]>;

class AE_MULAF16SS_00_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf16ss.00 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF16SS_00 : AE_MULAF16SS_00_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf16ss_00 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF16SS_00_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf16ss.00_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF16SS_00_S2 : AE_MULAF16SS_00_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf16ss_00_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF16SS_10_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaf16ss.10 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAF16SS_10 : AE_MULAF16SS_10_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaf16ss_10 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAF16SS_11_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaf16ss.11 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{38} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAF16SS_11 : AE_MULAF16SS_11_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaf16ss_11 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAF16SS_20_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaf16ss.20 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAF16SS_20 : AE_MULAF16SS_20_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaf16ss_20 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAF16SS_21_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaf16ss.21 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{38} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAF16SS_21 : AE_MULAF16SS_21_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaf16ss_21 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAF16SS_22_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaf16ss.22 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAF16SS_22 : AE_MULAF16SS_22_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaf16ss_22 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAF16SS_30_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaf16ss.30 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{38} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAF16SS_30 : AE_MULAF16SS_30_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaf16ss_30 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAF16SS_31_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaf16ss.31 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAF16SS_31 : AE_MULAF16SS_31_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaf16ss_31 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAF16SS_32_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaf16ss.32 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{36} = 1;
let Inst{38} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAF16SS_32 : AE_MULAF16SS_32_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaf16ss_32 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAF16SS_33_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulaf16ss.33 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{36} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAF16SS_33 : AE_MULAF16SS_33_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulaf16ss_33 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAF16X4SS_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q1_out, AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q1, AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d0), "ae_mulaf16x4ss $ae_mul_q1, $ae_mul_q0, $ae_mul_d1, $ae_mul_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q1;
bits<4> ae_mul_q0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d0;
let Constraints = "$ae_mul_q1 = $ae_mul_q1_out,@earlyclobber $ae_mul_q1_out, $ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
}



def AE_MULAF16X4SS : AE_MULAF16X4SS_AE_FORMAT2<[]>;

class AE_MULAF32R_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32r.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32R_HH : AE_MULAF32R_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32r_hh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32R_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32r.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32R_LH : AE_MULAF32R_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32r_lh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32R_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32r.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32R_LL : AE_MULAF32R_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32r_ll AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32R_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf32r.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF32R_LL_S2 : AE_MULAF32R_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf32r_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF32S_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32s.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32S_HH : AE_MULAF32S_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32s_hh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32S_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32s.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32S_LH : AE_MULAF32S_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32s_lh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32S_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32s.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32S_LL : AE_MULAF32S_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32s_ll AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32S_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf32s.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF32S_LL_S2 : AE_MULAF32S_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf32s_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF32X16_H0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32x16.h0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32X16_H0 : AE_MULAF32X16_H0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32x16_h0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32X16_H0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf32x16.h0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF32X16_H0_S2 : AE_MULAF32X16_H0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf32x16_h0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF32X16_H1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32x16.h1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32X16_H1 : AE_MULAF32X16_H1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32x16_h1 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32X16_H1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf32x16.h1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF32X16_H1_S2 : AE_MULAF32X16_H1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf32x16_h1_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF32X16_H2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32x16.h2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32X16_H2 : AE_MULAF32X16_H2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32x16_h2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32X16_H2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf32x16.h2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF32X16_H2_S2 : AE_MULAF32X16_H2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf32x16_h2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF32X16_H3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32x16.h3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32X16_H3 : AE_MULAF32X16_H3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32x16_h3 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32X16_H3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf32x16.h3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF32X16_H3_S2 : AE_MULAF32X16_H3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf32x16_h3_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF32X16_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32x16.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32X16_L0 : AE_MULAF32X16_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32x16_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32X16_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf32x16.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF32X16_L0_S2 : AE_MULAF32X16_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf32x16_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF32X16_L1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32x16.l1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32X16_L1 : AE_MULAF32X16_L1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32x16_l1 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32X16_L1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf32x16.l1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF32X16_L1_S2 : AE_MULAF32X16_L1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf32x16_l1_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF32X16_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32x16.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32X16_L2 : AE_MULAF32X16_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32x16_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32X16_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf32x16.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF32X16_L2_S2 : AE_MULAF32X16_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf32x16_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF32X16_L3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf32x16.l3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF32X16_L3 : AE_MULAF32X16_L3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf32x16_l3 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF32X16_L3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf32x16.l3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF32X16_L3_S2 : AE_MULAF32X16_L3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf32x16_l3_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF48Q32SP16S_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf48q32sp16s.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF48Q32SP16S_L : AE_MULAF48Q32SP16S_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf48q32sp16s_l AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF48Q32SP16S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf48q32sp16s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF48Q32SP16S_L_S2 : AE_MULAF48Q32SP16S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf48q32sp16s_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAF48Q32SP16U_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulaf48q32sp16u.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAF48Q32SP16U_L : AE_MULAF48Q32SP16U_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulaf48q32sp16u_l AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAF48Q32SP16U_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaf48q32sp16u.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAF48Q32SP16U_L_S2 : AE_MULAF48Q32SP16U_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaf48q32sp16u_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAFC24RA_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x4_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x4_q0, AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1), "ae_mulafc24ra $opnd_ae_sem_mul_x4_q0, $opnd_ae_sem_mul_x4_d0, $opnd_ae_sem_mul_x4_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x4_q0;
bits<4> opnd_ae_sem_mul_x4_d0;
bits<4> opnd_ae_sem_mul_x4_d1;
let Constraints = "$opnd_ae_sem_mul_x4_q0 = $opnd_ae_sem_mul_x4_q0_out,@earlyclobber $opnd_ae_sem_mul_x4_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x4_q0{0};
let Inst{17} = opnd_ae_sem_mul_x4_q0{1};
let Inst{18} = opnd_ae_sem_mul_x4_q0{2};
let Inst{19} = opnd_ae_sem_mul_x4_q0{3};
let Inst{24} = opnd_ae_sem_mul_x4_d0{0};
let Inst{25} = opnd_ae_sem_mul_x4_d0{1};
let Inst{26} = opnd_ae_sem_mul_x4_d0{2};
let Inst{27} = opnd_ae_sem_mul_x4_d0{3};
let Inst{20} = opnd_ae_sem_mul_x4_d1{0};
let Inst{21} = opnd_ae_sem_mul_x4_d1{1};
let Inst{22} = opnd_ae_sem_mul_x4_d1{2};
let Inst{23} = opnd_ae_sem_mul_x4_d1{3};
}



def AE_MULAFC24RA : AE_MULAFC24RA_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x4_q0_out, (int_xtensa_ae_mulafc24ra AE_DR:$opnd_ae_sem_mul_x4_q0, AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1))]>;

class AE_MULAFC32X16RAS_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x4_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x4_q0, AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1), "ae_mulafc32x16ras.h $opnd_ae_sem_mul_x4_q0, $opnd_ae_sem_mul_x4_d0, $opnd_ae_sem_mul_x4_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x4_q0;
bits<4> opnd_ae_sem_mul_x4_d0;
bits<4> opnd_ae_sem_mul_x4_d1;
let Constraints = "$opnd_ae_sem_mul_x4_q0 = $opnd_ae_sem_mul_x4_q0_out,@earlyclobber $opnd_ae_sem_mul_x4_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x4_q0{0};
let Inst{17} = opnd_ae_sem_mul_x4_q0{1};
let Inst{18} = opnd_ae_sem_mul_x4_q0{2};
let Inst{19} = opnd_ae_sem_mul_x4_q0{3};
let Inst{24} = opnd_ae_sem_mul_x4_d0{0};
let Inst{25} = opnd_ae_sem_mul_x4_d0{1};
let Inst{26} = opnd_ae_sem_mul_x4_d0{2};
let Inst{27} = opnd_ae_sem_mul_x4_d0{3};
let Inst{20} = opnd_ae_sem_mul_x4_d1{0};
let Inst{21} = opnd_ae_sem_mul_x4_d1{1};
let Inst{22} = opnd_ae_sem_mul_x4_d1{2};
let Inst{23} = opnd_ae_sem_mul_x4_d1{3};
}



def AE_MULAFC32X16RAS_H : AE_MULAFC32X16RAS_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x4_q0_out, (int_xtensa_ae_mulafc32x16ras_h AE_DR:$opnd_ae_sem_mul_x4_q0, AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1))]>;

class AE_MULAFC32X16RAS_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x4_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x4_q0, AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1), "ae_mulafc32x16ras.l $opnd_ae_sem_mul_x4_q0, $opnd_ae_sem_mul_x4_d0, $opnd_ae_sem_mul_x4_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x4_q0;
bits<4> opnd_ae_sem_mul_x4_d0;
bits<4> opnd_ae_sem_mul_x4_d1;
let Constraints = "$opnd_ae_sem_mul_x4_q0 = $opnd_ae_sem_mul_x4_q0_out,@earlyclobber $opnd_ae_sem_mul_x4_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x4_q0{0};
let Inst{17} = opnd_ae_sem_mul_x4_q0{1};
let Inst{18} = opnd_ae_sem_mul_x4_q0{2};
let Inst{19} = opnd_ae_sem_mul_x4_q0{3};
let Inst{24} = opnd_ae_sem_mul_x4_d0{0};
let Inst{25} = opnd_ae_sem_mul_x4_d0{1};
let Inst{26} = opnd_ae_sem_mul_x4_d0{2};
let Inst{27} = opnd_ae_sem_mul_x4_d0{3};
let Inst{20} = opnd_ae_sem_mul_x4_d1{0};
let Inst{21} = opnd_ae_sem_mul_x4_d1{1};
let Inst{22} = opnd_ae_sem_mul_x4_d1{2};
let Inst{23} = opnd_ae_sem_mul_x4_d1{3};
}



def AE_MULAFC32X16RAS_L : AE_MULAFC32X16RAS_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x4_q0_out, (int_xtensa_ae_mulafc32x16ras_l AE_DR:$opnd_ae_sem_mul_x4_q0, AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1))]>;

class AE_MULAFD24X2_FIR_H_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out, AE_DR:$ae_mul_q1_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulafd24x2.fir.h $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out, $ae_mul_q1 = $ae_mul_q1_out,@earlyclobber $ae_mul_q1_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode

//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULAFD24X2_FIR_H : AE_MULAFD24X2_FIR_H_AE_FORMAT2<[]>;

class AE_MULAFD24X2_FIR_L_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out, AE_DR:$ae_mul_q1_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulafd24x2.fir.l $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out, $ae_mul_q1 = $ae_mul_q1_out,@earlyclobber $ae_mul_q1_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULAFD24X2_FIR_L : AE_MULAFD24X2_FIR_L_AE_FORMAT2<[]>;

class AE_MULAFD32X16X2_FIR_HH_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out, AE_DR:$ae_mul_q1_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulafd32x16x2.fir.hh $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out, $ae_mul_q1 = $ae_mul_q1_out,@earlyclobber $ae_mul_q1_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULAFD32X16X2_FIR_HH : AE_MULAFD32X16X2_FIR_HH_AE_FORMAT2<[]>;

class AE_MULAFD32X16X2_FIR_HL_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out, AE_DR:$ae_mul_q1_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulafd32x16x2.fir.hl $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out, $ae_mul_q1 = $ae_mul_q1_out,@earlyclobber $ae_mul_q1_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULAFD32X16X2_FIR_HL : AE_MULAFD32X16X2_FIR_HL_AE_FORMAT2<[]>;

class AE_MULAFD32X16X2_FIR_LH_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out, AE_DR:$ae_mul_q1_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulafd32x16x2.fir.lh $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out, $ae_mul_q1 = $ae_mul_q1_out,@earlyclobber $ae_mul_q1_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULAFD32X16X2_FIR_LH : AE_MULAFD32X16X2_FIR_LH_AE_FORMAT2<[]>;

class AE_MULAFD32X16X2_FIR_LL_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out, AE_DR:$ae_mul_q1_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulafd32x16x2.fir.ll $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out, $ae_mul_q1 = $ae_mul_q1_out,@earlyclobber $ae_mul_q1_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULAFD32X16X2_FIR_LL : AE_MULAFD32X16X2_FIR_LL_AE_FORMAT2<[]>;

class AE_MULAFP24X2R_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulafp24x2r $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAFP24X2R : AE_MULAFP24X2R_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulafp24x2r AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAFP24X2R_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulafp24x2r_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAFP24X2R_S2 : AE_MULAFP24X2R_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulafp24x2r_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAFP24X2RA_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulafp24x2ra $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAFP24X2RA : AE_MULAFP24X2RA_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulafp24x2ra AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAFP24X2RA_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulafp24x2ra_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAFP24X2RA_S2 : AE_MULAFP24X2RA_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulafp24x2ra_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAFP32X16X2RAS_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulafp32x16x2ras.h $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAFP32X16X2RAS_H : AE_MULAFP32X16X2RAS_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulafp32x16x2ras_h AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAFP32X16X2RAS_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulafp32x16x2ras.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAFP32X16X2RAS_H_S2 : AE_MULAFP32X16X2RAS_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulafp32x16x2ras_h_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAFP32X16X2RAS_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulafp32x16x2ras.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAFP32X16X2RAS_L : AE_MULAFP32X16X2RAS_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulafp32x16x2ras_l AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAFP32X16X2RAS_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulafp32x16x2ras.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAFP32X16X2RAS_L_S2 : AE_MULAFP32X16X2RAS_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulafp32x16x2ras_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAFP32X16X2RS_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulafp32x16x2rs.h $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAFP32X16X2RS_H : AE_MULAFP32X16X2RS_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulafp32x16x2rs_h AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAFP32X16X2RS_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulafp32x16x2rs.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAFP32X16X2RS_H_S2 : AE_MULAFP32X16X2RS_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulafp32x16x2rs_h_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAFP32X16X2RS_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulafp32x16x2rs.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAFP32X16X2RS_L : AE_MULAFP32X16X2RS_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulafp32x16x2rs_l AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAFP32X16X2RS_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulafp32x16x2rs.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAFP32X16X2RS_L_S2 : AE_MULAFP32X16X2RS_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulafp32x16x2rs_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAFP32X2RAS_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulafp32x2ras $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAFP32X2RAS : AE_MULAFP32X2RAS_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulafp32x2ras AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAFP32X2RS_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulafp32x2rs $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAFP32X2RS : AE_MULAFP32X2RS_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulafp32x2rs AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAFQ32SP24S_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulafq32sp24s.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAFQ32SP24S_H_S2 : AE_MULAFQ32SP24S_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulafq32sp24s_h_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAFQ32SP24S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulafq32sp24s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAFQ32SP24S_L_S2 : AE_MULAFQ32SP24S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulafq32sp24s_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAP24X2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulap24x2 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{54} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULAP24X2 : AE_MULAP24X2_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulap24x2 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULAP24X2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulap24x2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAP24X2_S2 : AE_MULAP24X2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulap24x2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAP32X16X2_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulap32x16x2.h $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAP32X16X2_H : AE_MULAP32X16X2_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulap32x16x2_h AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAP32X16X2_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulap32x16x2.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAP32X16X2_L : AE_MULAP32X16X2_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulap32x16x2_l AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAP32X2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulap32x2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAP32X2 : AE_MULAP32X2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulap32x2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAQ32SP16S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaq32sp16s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAQ32SP16S_L_S2 : AE_MULAQ32SP16S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaq32sp16s_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAQ32SP16U_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulaq32sp16u.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAQ32SP16U_L_S2 : AE_MULAQ32SP16U_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulaq32sp16u_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULARFQ32SP24S_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mularfq32sp24s.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULARFQ32SP24S_H_S2 : AE_MULARFQ32SP24S_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mularfq32sp24s_h_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULARFQ32SP24S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mularfq32sp24s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULARFQ32SP24S_L_S2 : AE_MULARFQ32SP24S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mularfq32sp24s_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAS32F48P16S_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulas32f48p16s.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAS32F48P16S_HH : AE_MULAS32F48P16S_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulas32f48p16s_hh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAS32F48P16S_HH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulas32f48p16s.hh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAS32F48P16S_HH_S2 : AE_MULAS32F48P16S_HH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulas32f48p16s_hh_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAS32F48P16S_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulas32f48p16s.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAS32F48P16S_LH : AE_MULAS32F48P16S_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulas32f48p16s_lh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAS32F48P16S_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulas32f48p16s.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAS32F48P16S_LH_S2 : AE_MULAS32F48P16S_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulas32f48p16s_lh_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULAS32F48P16S_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulas32f48p16s.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULAS32F48P16S_LL : AE_MULAS32F48P16S_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulas32f48p16s_ll AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULAS32F48P16S_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulas32f48p16s.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULAS32F48P16S_LL_S2 : AE_MULAS32F48P16S_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulas32f48p16s_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULASD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulasd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULASD24_HH_LL : AE_MULASD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulasd24_hh_ll AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULASD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulasd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULASD24_HH_LL_S2 : AE_MULASD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulasd24_hh_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULASD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulasd24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULASD24_HL_LH : AE_MULASD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulasd24_hl_lh AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULASD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulasd24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULASD24_HL_LH_S2 : AE_MULASD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulasd24_hl_lh_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULASD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulasd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULASD32X16_H1_L0 : AE_MULASD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulasd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULASD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulasd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULASD32X16_H1_L0_S2 : AE_MULASD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulasd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULASD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulasd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULASD32X16_H3_L2 : AE_MULASD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulasd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULASD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulasd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULASD32X16_H3_L2_S2 : AE_MULASD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulasd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULASFD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulasfd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{50} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULASFD24_HH_LL : AE_MULASFD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulasfd24_hh_ll AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULASFD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulasfd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULASFD24_HH_LL_S2 : AE_MULASFD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulasfd24_hh_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULASFD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulasfd24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULASFD24_HL_LH : AE_MULASFD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulasfd24_hl_lh AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULASFD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulasfd24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULASFD24_HL_LH_S2 : AE_MULASFD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulasfd24_hl_lh_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULASFD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulasfd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULASFD32X16_H1_L0 : AE_MULASFD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulasfd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULASFD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulasfd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULASFD32X16_H1_L0_S2 : AE_MULASFD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulasfd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULASFD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulasfd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULASFD32X16_H3_L2 : AE_MULASFD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulasfd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULASFD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulasfd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULASFD32X16_H3_L2_S2 : AE_MULASFD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulasfd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULC24_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulc24 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULC24 : AE_MULC24_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulc24 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULC32X16_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x4_q0), (ins AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1), "ae_mulc32x16.h $opnd_ae_sem_mul_x4_q0, $opnd_ae_sem_mul_x4_d0, $opnd_ae_sem_mul_x4_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x4_q0;
bits<4> opnd_ae_sem_mul_x4_d0;
bits<4> opnd_ae_sem_mul_x4_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x4_q0{0};
let Inst{17} = opnd_ae_sem_mul_x4_q0{1};
let Inst{18} = opnd_ae_sem_mul_x4_q0{2};
let Inst{19} = opnd_ae_sem_mul_x4_q0{3};
let Inst{24} = opnd_ae_sem_mul_x4_d0{0};
let Inst{25} = opnd_ae_sem_mul_x4_d0{1};
let Inst{26} = opnd_ae_sem_mul_x4_d0{2};
let Inst{27} = opnd_ae_sem_mul_x4_d0{3};
let Inst{20} = opnd_ae_sem_mul_x4_d1{0};
let Inst{21} = opnd_ae_sem_mul_x4_d1{1};
let Inst{22} = opnd_ae_sem_mul_x4_d1{2};
let Inst{23} = opnd_ae_sem_mul_x4_d1{3};
}



def AE_MULC32X16_H : AE_MULC32X16_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x4_q0, (int_xtensa_ae_mulc32x16_h AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1))]>;

class AE_MULC32X16_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x4_q0), (ins AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1), "ae_mulc32x16.l $opnd_ae_sem_mul_x4_q0, $opnd_ae_sem_mul_x4_d0, $opnd_ae_sem_mul_x4_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x4_q0;
bits<4> opnd_ae_sem_mul_x4_d0;
bits<4> opnd_ae_sem_mul_x4_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x4_q0{0};
let Inst{17} = opnd_ae_sem_mul_x4_q0{1};
let Inst{18} = opnd_ae_sem_mul_x4_q0{2};
let Inst{19} = opnd_ae_sem_mul_x4_q0{3};
let Inst{24} = opnd_ae_sem_mul_x4_d0{0};
let Inst{25} = opnd_ae_sem_mul_x4_d0{1};
let Inst{26} = opnd_ae_sem_mul_x4_d0{2};
let Inst{27} = opnd_ae_sem_mul_x4_d0{3};
let Inst{20} = opnd_ae_sem_mul_x4_d1{0};
let Inst{21} = opnd_ae_sem_mul_x4_d1{1};
let Inst{22} = opnd_ae_sem_mul_x4_d1{2};
let Inst{23} = opnd_ae_sem_mul_x4_d1{3};
}



def AE_MULC32X16_L : AE_MULC32X16_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x4_q0, (int_xtensa_ae_mulc32x16_l AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1))]>;

class AE_MULF16SS_00_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf16ss.00 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF16SS_00 : AE_MULF16SS_00_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf16ss_00 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF16SS_00_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf16ss.00_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF16SS_00_S2 : AE_MULF16SS_00_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf16ss_00_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF16SS_10_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulf16ss.10 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULF16SS_10 : AE_MULF16SS_10_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulf16ss_10 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULF16SS_11_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulf16ss.11 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULF16SS_11 : AE_MULF16SS_11_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulf16ss_11 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULF16SS_20_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulf16ss.20 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULF16SS_20 : AE_MULF16SS_20_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulf16ss_20 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULF16SS_21_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulf16ss.21 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULF16SS_21 : AE_MULF16SS_21_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulf16ss_21 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULF16SS_22_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulf16ss.22 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULF16SS_22 : AE_MULF16SS_22_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulf16ss_22 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULF16SS_30_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulf16ss.30 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULF16SS_30 : AE_MULF16SS_30_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulf16ss_30 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULF16SS_31_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulf16ss.31 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULF16SS_31 : AE_MULF16SS_31_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulf16ss_31 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULF16SS_32_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulf16ss.32 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULF16SS_32 : AE_MULF16SS_32_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulf16ss_32 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULF16SS_33_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulf16ss.33 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULF16SS_33 : AE_MULF16SS_33_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulf16ss_33 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULF16X4SS_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q1, AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d0), "ae_mulf16x4ss $ae_mul_q1, $ae_mul_q0, $ae_mul_d1, $ae_mul_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q1;
bits<4> ae_mul_q0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d0;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
}



def AE_MULF16X4SS : AE_MULF16X4SS_AE_FORMAT2<[]>;

class AE_MULF32R_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32r.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32R_HH : AE_MULF32R_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32r_hh AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32R_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32r.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32R_LH : AE_MULF32R_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32r_lh AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32R_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32r.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32R_LL : AE_MULF32R_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32r_ll AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32R_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf32r.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF32R_LL_S2 : AE_MULF32R_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf32r_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF32S_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32s.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32S_HH : AE_MULF32S_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32s_hh AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32S_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32s.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32S_LH : AE_MULF32S_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32s_lh AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32S_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32s.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32S_LL : AE_MULF32S_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32s_ll AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32S_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf32s.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF32S_LL_S2 : AE_MULF32S_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf32s_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF32X16_H0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32x16.h0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32X16_H0 : AE_MULF32X16_H0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32x16_h0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32X16_H0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf32x16.h0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF32X16_H0_S2 : AE_MULF32X16_H0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf32x16_h0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF32X16_H1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32x16.h1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32X16_H1 : AE_MULF32X16_H1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32x16_h1 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32X16_H1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf32x16.h1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF32X16_H1_S2 : AE_MULF32X16_H1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf32x16_h1_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF32X16_H2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32x16.h2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32X16_H2 : AE_MULF32X16_H2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32x16_h2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32X16_H2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf32x16.h2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF32X16_H2_S2 : AE_MULF32X16_H2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf32x16_h2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF32X16_H3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32x16.h3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32X16_H3 : AE_MULF32X16_H3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32x16_h3 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32X16_H3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf32x16.h3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF32X16_H3_S2 : AE_MULF32X16_H3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf32x16_h3_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF32X16_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32x16.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32X16_L0 : AE_MULF32X16_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32x16_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32X16_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf32x16.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF32X16_L0_S2 : AE_MULF32X16_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf32x16_l0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF32X16_L1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32x16.l1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32X16_L1 : AE_MULF32X16_L1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32x16_l1 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32X16_L1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf32x16.l1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF32X16_L1_S2 : AE_MULF32X16_L1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf32x16_l1_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF32X16_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32x16.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32X16_L2 : AE_MULF32X16_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32x16_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32X16_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf32x16.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF32X16_L2_S2 : AE_MULF32X16_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf32x16_l2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF32X16_L3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf32x16.l3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF32X16_L3 : AE_MULF32X16_L3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf32x16_l3 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF32X16_L3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf32x16.l3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF32X16_L3_S2 : AE_MULF32X16_L3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf32x16_l3_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF48Q32SP16S_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf48q32sp16s.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF48Q32SP16S_L : AE_MULF48Q32SP16S_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf48q32sp16s_l AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF48Q32SP16S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf48q32sp16s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF48Q32SP16S_L_S2 : AE_MULF48Q32SP16S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf48q32sp16s_l_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULF48Q32SP16U_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulf48q32sp16u.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULF48Q32SP16U_L : AE_MULF48Q32SP16U_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulf48q32sp16u_l AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULF48Q32SP16U_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulf48q32sp16u.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULF48Q32SP16U_L_S2 : AE_MULF48Q32SP16U_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulf48q32sp16u_l_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULFC24RA_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x4_q0), (ins AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1), "ae_mulfc24ra $opnd_ae_sem_mul_x4_q0, $opnd_ae_sem_mul_x4_d0, $opnd_ae_sem_mul_x4_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x4_q0;
bits<4> opnd_ae_sem_mul_x4_d0;
bits<4> opnd_ae_sem_mul_x4_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x4_q0{0};
let Inst{17} = opnd_ae_sem_mul_x4_q0{1};
let Inst{18} = opnd_ae_sem_mul_x4_q0{2};
let Inst{19} = opnd_ae_sem_mul_x4_q0{3};
let Inst{24} = opnd_ae_sem_mul_x4_d0{0};
let Inst{25} = opnd_ae_sem_mul_x4_d0{1};
let Inst{26} = opnd_ae_sem_mul_x4_d0{2};
let Inst{27} = opnd_ae_sem_mul_x4_d0{3};
let Inst{20} = opnd_ae_sem_mul_x4_d1{0};
let Inst{21} = opnd_ae_sem_mul_x4_d1{1};
let Inst{22} = opnd_ae_sem_mul_x4_d1{2};
let Inst{23} = opnd_ae_sem_mul_x4_d1{3};
}



def AE_MULFC24RA : AE_MULFC24RA_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x4_q0, (int_xtensa_ae_mulfc24ra AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1))]>;

class AE_MULFC32X16RAS_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x4_q0), (ins AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1), "ae_mulfc32x16ras.h $opnd_ae_sem_mul_x4_q0, $opnd_ae_sem_mul_x4_d0, $opnd_ae_sem_mul_x4_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x4_q0;
bits<4> opnd_ae_sem_mul_x4_d0;
bits<4> opnd_ae_sem_mul_x4_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x4_q0{0};
let Inst{17} = opnd_ae_sem_mul_x4_q0{1};
let Inst{18} = opnd_ae_sem_mul_x4_q0{2};
let Inst{19} = opnd_ae_sem_mul_x4_q0{3};
let Inst{24} = opnd_ae_sem_mul_x4_d0{0};
let Inst{25} = opnd_ae_sem_mul_x4_d0{1};
let Inst{26} = opnd_ae_sem_mul_x4_d0{2};
let Inst{27} = opnd_ae_sem_mul_x4_d0{3};
let Inst{20} = opnd_ae_sem_mul_x4_d1{0};
let Inst{21} = opnd_ae_sem_mul_x4_d1{1};
let Inst{22} = opnd_ae_sem_mul_x4_d1{2};
let Inst{23} = opnd_ae_sem_mul_x4_d1{3};
}



def AE_MULFC32X16RAS_H : AE_MULFC32X16RAS_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x4_q0, (int_xtensa_ae_mulfc32x16ras_h AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1))]>;

class AE_MULFC32X16RAS_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x4_q0), (ins AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1), "ae_mulfc32x16ras.l $opnd_ae_sem_mul_x4_q0, $opnd_ae_sem_mul_x4_d0, $opnd_ae_sem_mul_x4_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x4_q0;
bits<4> opnd_ae_sem_mul_x4_d0;
bits<4> opnd_ae_sem_mul_x4_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x4_q0{0};
let Inst{17} = opnd_ae_sem_mul_x4_q0{1};
let Inst{18} = opnd_ae_sem_mul_x4_q0{2};
let Inst{19} = opnd_ae_sem_mul_x4_q0{3};
let Inst{24} = opnd_ae_sem_mul_x4_d0{0};
let Inst{25} = opnd_ae_sem_mul_x4_d0{1};
let Inst{26} = opnd_ae_sem_mul_x4_d0{2};
let Inst{27} = opnd_ae_sem_mul_x4_d0{3};
let Inst{20} = opnd_ae_sem_mul_x4_d1{0};
let Inst{21} = opnd_ae_sem_mul_x4_d1{1};
let Inst{22} = opnd_ae_sem_mul_x4_d1{2};
let Inst{23} = opnd_ae_sem_mul_x4_d1{3};
}



def AE_MULFC32X16RAS_L : AE_MULFC32X16RAS_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x4_q0, (int_xtensa_ae_mulfc32x16ras_l AE_DR:$opnd_ae_sem_mul_x4_d0, AE_DR:$opnd_ae_sem_mul_x4_d1))]>;

class AE_MULFD24X2_FIR_H_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulfd24x2.fir.h $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULFD24X2_FIR_H : AE_MULFD24X2_FIR_H_AE_FORMAT2<[]>;

class AE_MULFD24X2_FIR_L_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulfd24x2.fir.l $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULFD24X2_FIR_L : AE_MULFD24X2_FIR_L_AE_FORMAT2<[]>;

class AE_MULFD32X16X2_FIR_HH_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulfd32x16x2.fir.hh $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULFD32X16X2_FIR_HH : AE_MULFD32X16X2_FIR_HH_AE_FORMAT2<[]>;

class AE_MULFD32X16X2_FIR_HL_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulfd32x16x2.fir.hl $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULFD32X16X2_FIR_HL : AE_MULFD32X16X2_FIR_HL_AE_FORMAT2<[]>;

class AE_MULFD32X16X2_FIR_LH_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulfd32x16x2.fir.lh $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULFD32X16X2_FIR_LH : AE_MULFD32X16X2_FIR_LH_AE_FORMAT2<[]>;

class AE_MULFD32X16X2_FIR_LL_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0, AE_DR:$ae_mul_q1), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d2), "ae_mulfd32x16x2.fir.ll $ae_mul_q0, $ae_mul_q1, $ae_mul_d0, $ae_mul_d1, $ae_mul_d2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_q1;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d2;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{36} = ae_mul_d2{0};
let Inst{37} = ae_mul_d2{1};
let Inst{38} = ae_mul_d2{2};
let Inst{39} = ae_mul_d2{3};
}



def AE_MULFD32X16X2_FIR_LL : AE_MULFD32X16X2_FIR_LL_AE_FORMAT2<[]>;

class AE_MULFP16X4RAS_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulfp16x4ras $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{21} = 1;
let Inst{23} = 1;
let Inst{36} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULFP16X4RAS : AE_MULFP16X4RAS_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulfp16x4ras AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULFP16X4S_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulfp16x4s $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{21} = 1;
let Inst{23} = 1;
let Inst{36} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULFP16X4S : AE_MULFP16X4S_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulfp16x4s AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULFP24X2R_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulfp24x2r $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULFP24X2R : AE_MULFP24X2R_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulfp24x2r AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULFP24X2R_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulfp24x2r_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULFP24X2R_S2 : AE_MULFP24X2R_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulfp24x2r_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULFP24X2RA_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulfp24x2ra $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULFP24X2RA : AE_MULFP24X2RA_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulfp24x2ra AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULFP24X2RA_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulfp24x2ra_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULFP24X2RA_S2 : AE_MULFP24X2RA_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulfp24x2ra_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULFP32X16X2RAS_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulfp32x16x2ras.h $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULFP32X16X2RAS_H : AE_MULFP32X16X2RAS_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulfp32x16x2ras_h AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULFP32X16X2RAS_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulfp32x16x2ras.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULFP32X16X2RAS_H_S2 : AE_MULFP32X16X2RAS_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulfp32x16x2ras_h_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULFP32X16X2RAS_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulfp32x16x2ras.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULFP32X16X2RAS_L : AE_MULFP32X16X2RAS_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulfp32x16x2ras_l AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULFP32X16X2RAS_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulfp32x16x2ras.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULFP32X16X2RAS_L_S2 : AE_MULFP32X16X2RAS_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulfp32x16x2ras_l_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULFP32X16X2RS_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulfp32x16x2rs.h $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULFP32X16X2RS_H : AE_MULFP32X16X2RS_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulfp32x16x2rs_h AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULFP32X16X2RS_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulfp32x16x2rs.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULFP32X16X2RS_H_S2 : AE_MULFP32X16X2RS_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulfp32x16x2rs_h_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULFP32X16X2RS_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulfp32x16x2rs.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULFP32X16X2RS_L : AE_MULFP32X16X2RS_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulfp32x16x2rs_l AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULFP32X16X2RS_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulfp32x16x2rs.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULFP32X16X2RS_L_S2 : AE_MULFP32X16X2RS_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulfp32x16x2rs_l_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULFP32X2RAS_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulfp32x2ras $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULFP32X2RAS : AE_MULFP32X2RAS_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulfp32x2ras AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULFP32X2RS_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulfp32x2rs $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULFP32X2RS : AE_MULFP32X2RS_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulfp32x2rs AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULFQ32SP24S_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulfq32sp24s.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULFQ32SP24S_H_S2 : AE_MULFQ32SP24S_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulfq32sp24s_h_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULFQ32SP24S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulfq32sp24s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULFQ32SP24S_L_S2 : AE_MULFQ32SP24S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulfq32sp24s_l_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULP24X2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulp24x2 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULP24X2 : AE_MULP24X2_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulp24x2 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULP24X2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulp24x2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULP24X2_S2 : AE_MULP24X2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulp24x2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULP32X16X2_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulp32x16x2.h $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULP32X16X2_H : AE_MULP32X16X2_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulp32x16x2_h AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULP32X16X2_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulp32x16x2.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULP32X16X2_L : AE_MULP32X16X2_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulp32x16x2_l AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULP32X2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulp32x2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULP32X2 : AE_MULP32X2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulp32x2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULQ32SP16S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulq32sp16s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULQ32SP16S_L_S2 : AE_MULQ32SP16S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulq32sp16s_l_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULQ32SP16U_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulq32sp16u.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULQ32SP16U_L_S2 : AE_MULQ32SP16U_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulq32sp16u_l_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULRFQ32SP24S_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulrfq32sp24s.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULRFQ32SP24S_H_S2 : AE_MULRFQ32SP24S_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulrfq32sp24s_h_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULRFQ32SP24S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulrfq32sp24s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULRFQ32SP24S_L_S2 : AE_MULRFQ32SP24S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulrfq32sp24s_l_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS16X4_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q1_out, AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q1, AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d0), "ae_muls16x4 $ae_mul_q1, $ae_mul_q0, $ae_mul_d1, $ae_mul_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q1;
bits<4> ae_mul_q0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d0;
let Constraints = "$ae_mul_q1 = $ae_mul_q1_out,@earlyclobber $ae_mul_q1_out, $ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{37} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
}



def AE_MULS16X4 : AE_MULS16X4_AE_FORMAT2<[]>;

class AE_MULS32_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32_HH : AE_MULS32_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32_hh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32_LH : AE_MULS32_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32_lh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32_LL : AE_MULS32_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32_ll AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32F48P16S_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32f48p16s.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32F48P16S_HH : AE_MULS32F48P16S_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_muls32f48p16s_hh AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32F48P16S_HH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32f48p16s.hh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32F48P16S_HH_S2 : AE_MULS32F48P16S_HH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_muls32f48p16s_hh_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS32F48P16S_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32f48p16s.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32F48P16S_LH : AE_MULS32F48P16S_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_muls32f48p16s_lh AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32F48P16S_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32f48p16s.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32F48P16S_LH_S2 : AE_MULS32F48P16S_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_muls32f48p16s_lh_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS32F48P16S_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32f48p16s.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32F48P16S_LL : AE_MULS32F48P16S_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_muls32f48p16s_ll AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32F48P16S_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32f48p16s.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32F48P16S_LL_S2 : AE_MULS32F48P16S_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_muls32f48p16s_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS32U_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32u.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32U_LL : AE_MULS32U_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32u_ll AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32X16_H0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32x16.h0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32X16_H0 : AE_MULS32X16_H0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32x16_h0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32X16_H0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32x16.h0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32X16_H0_S2 : AE_MULS32X16_H0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_muls32x16_h0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS32X16_H1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32x16.h1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32X16_H1 : AE_MULS32X16_H1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32x16_h1 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32X16_H1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32x16.h1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32X16_H1_S2 : AE_MULS32X16_H1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_muls32x16_h1_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS32X16_H2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32x16.h2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32X16_H2 : AE_MULS32X16_H2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32x16_h2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32X16_H2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32x16.h2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32X16_H2_S2 : AE_MULS32X16_H2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_muls32x16_h2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS32X16_H3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32x16.h3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32X16_H3 : AE_MULS32X16_H3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32x16_h3 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32X16_H3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32x16.h3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32X16_H3_S2 : AE_MULS32X16_H3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_muls32x16_h3_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS32X16_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32x16.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32X16_L0 : AE_MULS32X16_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32x16_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32X16_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32x16.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32X16_L0_S2 : AE_MULS32X16_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_muls32x16_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS32X16_L1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32x16.l1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32X16_L1 : AE_MULS32X16_L1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32x16_l1 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32X16_L1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32x16.l1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32X16_L1_S2 : AE_MULS32X16_L1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_muls32x16_l1_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS32X16_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32x16.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32X16_L2 : AE_MULS32X16_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32x16_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32X16_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32x16.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32X16_L2_S2 : AE_MULS32X16_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_muls32x16_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULS32X16_L3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_muls32x16.l3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULS32X16_L3 : AE_MULS32X16_L3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_muls32x16_l3 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULS32X16_L3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_muls32x16.l3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULS32X16_L3_S2 : AE_MULS32X16_L3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_muls32x16_l3_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSAD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsad24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSAD24_HH_LL : AE_MULSAD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsad24_hh_ll AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSAD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsad24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSAD24_HH_LL_S2 : AE_MULSAD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsad24_hh_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSAD32X16_H1_L0_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsad32x16.h1.l0 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSAD32X16_H1_L0 : AE_MULSAD32X16_H1_L0_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsad32x16_h1_l0 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSAD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsad32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSAD32X16_H1_L0_S2 : AE_MULSAD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsad32x16_h1_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSAD32X16_H3_L2_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsad32x16.h3.l2 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{39} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSAD32X16_H3_L2 : AE_MULSAD32X16_H3_L2_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsad32x16_h3_l2 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSAD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsad32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSAD32X16_H3_L2_S2 : AE_MULSAD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsad32x16_h3_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSAFD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsafd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSAFD24_HH_LL : AE_MULSAFD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsafd24_hh_ll AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSAFD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsafd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSAFD24_HH_LL_S2 : AE_MULSAFD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsafd24_hh_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSAFD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsafd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSAFD32X16_H1_L0 : AE_MULSAFD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsafd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSAFD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsafd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSAFD32X16_H1_L0_S2 : AE_MULSAFD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsafd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSAFD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsafd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSAFD32X16_H3_L2 : AE_MULSAFD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsafd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSAFD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsafd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSAFD32X16_H3_L2_S2 : AE_MULSAFD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsafd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF16SS_00_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf16ss.00 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF16SS_00 : AE_MULSF16SS_00_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf16ss_00 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF16SS_00_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf16ss.00_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF16SS_00_S2 : AE_MULSF16SS_00_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf16ss_00_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF16SS_10_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsf16ss.10 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSF16SS_10 : AE_MULSF16SS_10_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsf16ss_10 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSF16SS_11_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsf16ss.11 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
let Inst{39} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSF16SS_11 : AE_MULSF16SS_11_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsf16ss_11 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSF16SS_20_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsf16ss.20 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSF16SS_20 : AE_MULSF16SS_20_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsf16ss_20 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSF16SS_21_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsf16ss.21 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{39} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSF16SS_21 : AE_MULSF16SS_21_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsf16ss_21 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSF16SS_22_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsf16ss.22 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{36} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSF16SS_22 : AE_MULSF16SS_22_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsf16ss_22 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSF16SS_30_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsf16ss.30 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{36} = 1;
let Inst{39} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSF16SS_30 : AE_MULSF16SS_30_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsf16ss_30 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSF16SS_31_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsf16ss.31 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{21} = 1;
let Inst{36} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSF16SS_31 : AE_MULSF16SS_31_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsf16ss_31 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSF16SS_32_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsf16ss.32 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{21} = 1;
let Inst{36} = 1;
let Inst{39} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSF16SS_32 : AE_MULSF16SS_32_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsf16ss_32 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSF16SS_33_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsf16ss.33 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{36} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSF16SS_33 : AE_MULSF16SS_33_AE_FORMAT2<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsf16ss_33 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSF16X4SS_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q1_out, AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q1, AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d1, AE_DR:$ae_mul_d0), "ae_mulsf16x4ss $ae_mul_q1, $ae_mul_q0, $ae_mul_d1, $ae_mul_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q1;
bits<4> ae_mul_q0;
bits<4> ae_mul_d1;
bits<4> ae_mul_d0;
let Constraints = "$ae_mul_q1 = $ae_mul_q1_out,@earlyclobber $ae_mul_q1_out, $ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{37} = 1;
let Inst{39} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{20} = ae_mul_q1{0};
let Inst{21} = ae_mul_q1{1};
let Inst{22} = ae_mul_q1{2};
let Inst{23} = ae_mul_q1{3};
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
}



def AE_MULSF16X4SS : AE_MULSF16X4SS_AE_FORMAT2<[]>;

class AE_MULSF32R_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32r.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32R_HH : AE_MULSF32R_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32r_hh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32R_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32r.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32R_LH : AE_MULSF32R_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32r_lh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32R_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32r.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32R_LL : AE_MULSF32R_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32r_ll AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32R_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf32r.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF32R_LL_S2 : AE_MULSF32R_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf32r_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF32S_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32s.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32S_HH : AE_MULSF32S_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32s_hh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32S_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32s.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32S_LH : AE_MULSF32S_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32s_lh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32S_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32s.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32S_LL : AE_MULSF32S_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32s_ll AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32X16_H0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32x16.h0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32X16_H0 : AE_MULSF32X16_H0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32x16_h0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32X16_H0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf32x16.h0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF32X16_H0_S2 : AE_MULSF32X16_H0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf32x16_h0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF32X16_H1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32x16.h1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32X16_H1 : AE_MULSF32X16_H1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32x16_h1 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32X16_H1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf32x16.h1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF32X16_H1_S2 : AE_MULSF32X16_H1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf32x16_h1_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF32X16_H2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32x16.h2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32X16_H2 : AE_MULSF32X16_H2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32x16_h2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32X16_H2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf32x16.h2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF32X16_H2_S2 : AE_MULSF32X16_H2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf32x16_h2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF32X16_H3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32x16.h3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32X16_H3 : AE_MULSF32X16_H3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32x16_h3 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32X16_H3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf32x16.h3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF32X16_H3_S2 : AE_MULSF32X16_H3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf32x16_h3_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF32X16_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32x16.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32X16_L0 : AE_MULSF32X16_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32x16_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32X16_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf32x16.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF32X16_L0_S2 : AE_MULSF32X16_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf32x16_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF32X16_L1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32x16.l1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32X16_L1 : AE_MULSF32X16_L1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32x16_l1 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32X16_L1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf32x16.l1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF32X16_L1_S2 : AE_MULSF32X16_L1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf32x16_l1_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF32X16_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32x16.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32X16_L2 : AE_MULSF32X16_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32x16_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32X16_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf32x16.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF32X16_L2_S2 : AE_MULSF32X16_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf32x16_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF32X16_L3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf32x16.l3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF32X16_L3 : AE_MULSF32X16_L3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf32x16_l3 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF32X16_L3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf32x16.l3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF32X16_L3_S2 : AE_MULSF32X16_L3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf32x16_l3_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF48Q32SP16S_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf48q32sp16s.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF48Q32SP16S_L : AE_MULSF48Q32SP16S_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf48q32sp16s_l AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF48Q32SP16S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf48q32sp16s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF48Q32SP16S_L_S2 : AE_MULSF48Q32SP16S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf48q32sp16s_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSF48Q32SP16U_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsf48q32sp16u.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSF48Q32SP16U_L : AE_MULSF48Q32SP16U_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsf48q32sp16u_l AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSF48Q32SP16U_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsf48q32sp16u.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSF48Q32SP16U_L_S2 : AE_MULSF48Q32SP16U_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsf48q32sp16u_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSFP24X2R_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsfp24x2r $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSFP24X2R : AE_MULSFP24X2R_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsfp24x2r AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSFP24X2R_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsfp24x2r_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSFP24X2R_S2 : AE_MULSFP24X2R_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsfp24x2r_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSFP24X2RA_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsfp24x2ra $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSFP24X2RA : AE_MULSFP24X2RA_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsfp24x2ra AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSFP24X2RA_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsfp24x2ra_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSFP24X2RA_S2 : AE_MULSFP24X2RA_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsfp24x2ra_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSFP32X16X2RAS_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsfp32x16x2ras.h $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSFP32X16X2RAS_H : AE_MULSFP32X16X2RAS_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsfp32x16x2ras_h AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSFP32X16X2RAS_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsfp32x16x2ras.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSFP32X16X2RAS_H_S2 : AE_MULSFP32X16X2RAS_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsfp32x16x2ras_h_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSFP32X16X2RAS_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsfp32x16x2ras.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSFP32X16X2RAS_L : AE_MULSFP32X16X2RAS_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsfp32x16x2ras_l AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSFP32X16X2RAS_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsfp32x16x2ras.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSFP32X16X2RAS_L_S2 : AE_MULSFP32X16X2RAS_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsfp32x16x2ras_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSFP32X16X2RS_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsfp32x16x2rs.h $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSFP32X16X2RS_H : AE_MULSFP32X16X2RS_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsfp32x16x2rs_h AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSFP32X16X2RS_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsfp32x16x2rs.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSFP32X16X2RS_H_S2 : AE_MULSFP32X16X2RS_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsfp32x16x2rs_h_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSFP32X16X2RS_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsfp32x16x2rs.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSFP32X16X2RS_L : AE_MULSFP32X16X2RS_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsfp32x16x2rs_l AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSFP32X16X2RS_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsfp32x16x2rs.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSFP32X16X2RS_L_S2 : AE_MULSFP32X16X2RS_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsfp32x16x2rs_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSFP32X2RAS_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsfp32x2ras $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSFP32X2RAS : AE_MULSFP32X2RAS_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsfp32x2ras AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSFP32X2RS_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsfp32x2rs $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSFP32X2RS : AE_MULSFP32X2RS_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsfp32x2rs AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSFQ32SP24S_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsfq32sp24s.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSFQ32SP24S_H_S2 : AE_MULSFQ32SP24S_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsfq32sp24s_h_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSFQ32SP24S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsfq32sp24s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSFQ32SP24S_L_S2 : AE_MULSFQ32SP24S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsfq32sp24s_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSP24X2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulsp24x2 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{53} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSP24X2 : AE_MULSP24X2_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulsp24x2 AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSP24X2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsp24x2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSP24X2_S2 : AE_MULSP24X2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsp24x2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSP32X16X2_H_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsp32x16x2.h $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSP32X16X2_H : AE_MULSP32X16X2_H_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsp32x16x2_h AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSP32X16X2_L_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsp32x16x2.l $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSP32X16X2_L : AE_MULSP32X16X2_L_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsp32x16x2_l AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSP32X2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulsp32x2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSP32X2 : AE_MULSP32X2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulsp32x2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSQ32SP16S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsq32sp16s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSQ32SP16S_L_S2 : AE_MULSQ32SP16S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsq32sp16s_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSQ32SP16U_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsq32sp16u.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSQ32SP16U_L_S2 : AE_MULSQ32SP16U_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsq32sp16u_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSRFQ32SP24S_H_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsrfq32sp24s.h_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSRFQ32SP24S_H_S2 : AE_MULSRFQ32SP24S_H_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsrfq32sp24s_h_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSRFQ32SP24S_L_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulsrfq32sp24s.l_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSRFQ32SP24S_L_S2 : AE_MULSRFQ32SP24S_L_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulsrfq32sp24s_l_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSS32F48P16S_HH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulss32f48p16s.hh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSS32F48P16S_HH : AE_MULSS32F48P16S_HH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulss32f48p16s_hh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSS32F48P16S_HH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulss32f48p16s.hh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSS32F48P16S_HH_S2 : AE_MULSS32F48P16S_HH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulss32f48p16s_hh_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSS32F48P16S_LH_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulss32f48p16s.lh $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSS32F48P16S_LH : AE_MULSS32F48P16S_LH_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulss32f48p16s_lh AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSS32F48P16S_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulss32f48p16s.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSS32F48P16S_LH_S2 : AE_MULSS32F48P16S_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulss32f48p16s_lh_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSS32F48P16S_LL_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulss32f48p16s.ll $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSS32F48P16S_LL : AE_MULSS32F48P16S_LL_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulss32f48p16s_ll AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSS32F48P16S_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulss32f48p16s.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSS32F48P16S_LL_S2 : AE_MULSS32F48P16S_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulss32f48p16s_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulssd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSSD24_HH_LL : AE_MULSSD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulssd24_hh_ll AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSSD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSD24_HH_LL_S2 : AE_MULSSD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssd24_hh_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulssd24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{50} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSSD24_HL_LH : AE_MULSSD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulssd24_hl_lh AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSSD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssd24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSD24_HL_LH_S2 : AE_MULSSD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssd24_hl_lh_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulssd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSSD32X16_H1_L0 : AE_MULSSD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulssd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSSD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSD32X16_H1_L0_S2 : AE_MULSSD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulssd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSSD32X16_H3_L2 : AE_MULSSD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulssd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSSD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSD32X16_H3_L2_S2 : AE_MULSSD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSFD16SS_11_00_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulssfd16ss.11_00 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSSFD16SS_11_00 : AE_MULSSFD16SS_11_00_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulssfd16ss_11_00 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSSFD16SS_11_00_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssfd16ss.11_00_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSFD16SS_11_00_S2 : AE_MULSSFD16SS_11_00_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssfd16ss_11_00_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSFD16SS_13_02_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulssfd16ss.13_02 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSSFD16SS_13_02 : AE_MULSSFD16SS_13_02_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulssfd16ss_13_02 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSSFD16SS_13_02_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssfd16ss.13_02_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSFD16SS_13_02_S2 : AE_MULSSFD16SS_13_02_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssfd16ss_13_02_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSFD16SS_33_22_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulssfd16ss.33_22 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSSFD16SS_33_22 : AE_MULSSFD16SS_33_22_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulssfd16ss_33_22 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSSFD16SS_33_22_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssfd16ss.33_22_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSFD16SS_33_22_S2 : AE_MULSSFD16SS_33_22_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssfd16ss_33_22_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSFD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulssfd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSSFD24_HH_LL : AE_MULSSFD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulssfd24_hh_ll AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSSFD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssfd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSFD24_HH_LL_S2 : AE_MULSSFD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssfd24_hh_ll_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSFD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0_out), (ins AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulssfd24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;
let Constraints = "$ae_mul_q0 = $ae_mul_q0_out,@earlyclobber $ae_mul_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULSSFD24_HL_LH : AE_MULSSFD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0_out, (int_xtensa_ae_mulssfd24_hl_lh AE_DR:$ae_mul_q0, AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULSSFD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssfd24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSFD24_HL_LH_S2 : AE_MULSSFD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssfd24_hl_lh_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSFD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulssfd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSSFD32X16_H1_L0 : AE_MULSSFD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulssfd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSSFD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssfd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSFD32X16_H1_L0_S2 : AE_MULSSFD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssfd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULSSFD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulssfd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;
let Constraints = "$opnd_ae_sem_mul_x2_S1_q0 = $opnd_ae_sem_mul_x2_S1_q0_out,@earlyclobber $opnd_ae_sem_mul_x2_S1_q0_out";


//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULSSFD32X16_H3_L2 : AE_MULSSFD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0_out, (int_xtensa_ae_mulssfd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_q0, AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULSSFD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0_out), (ins AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulssfd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;
let Constraints = "$ae_mul_S2_q0 = $ae_mul_S2_q0_out,@earlyclobber $ae_mul_S2_q0_out";


//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULSSFD32X16_H3_L2_S2 : AE_MULSSFD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0_out, (int_xtensa_ae_mulssfd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_q0, AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzaad24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZAAD24_HH_LL : AE_MULZAAD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzaad24_hh_ll AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZAAD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaad24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAD24_HH_LL_S2 : AE_MULZAAD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaad24_hh_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzaad24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZAAD24_HL_LH : AE_MULZAAD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzaad24_hl_lh AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZAAD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaad24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAD24_HL_LH_S2 : AE_MULZAAD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaad24_hl_lh_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAD32X16_H0_L1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaad32x16.h0.l1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAD32X16_H0_L1 : AE_MULZAAD32X16_H0_L1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaad32x16_h0_l1 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAD32X16_H0_L1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaad32x16.h0.l1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAD32X16_H0_L1_S2 : AE_MULZAAD32X16_H0_L1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaad32x16_h0_l1_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaad32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAD32X16_H1_L0 : AE_MULZAAD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaad32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaad32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAD32X16_H1_L0_S2 : AE_MULZAAD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaad32x16_h1_l0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAD32X16_H2_L3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaad32x16.h2.l3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAD32X16_H2_L3 : AE_MULZAAD32X16_H2_L3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaad32x16_h2_l3 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAD32X16_H2_L3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaad32x16.h2.l3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAD32X16_H2_L3_S2 : AE_MULZAAD32X16_H2_L3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaad32x16_h2_l3_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaad32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAD32X16_H3_L2 : AE_MULZAAD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaad32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaad32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAD32X16_H3_L2_S2 : AE_MULZAAD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaad32x16_h3_l2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAFD16SS_11_00_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaafd16ss.11_00 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAFD16SS_11_00 : AE_MULZAAFD16SS_11_00_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaafd16ss_11_00 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAFD16SS_11_00_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaafd16ss.11_00_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAFD16SS_11_00_S2 : AE_MULZAAFD16SS_11_00_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaafd16ss_11_00_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAFD16SS_13_02_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaafd16ss.13_02 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAFD16SS_13_02 : AE_MULZAAFD16SS_13_02_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaafd16ss_13_02 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAFD16SS_13_02_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaafd16ss.13_02_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAFD16SS_13_02_S2 : AE_MULZAAFD16SS_13_02_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaafd16ss_13_02_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAFD16SS_33_22_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaafd16ss.33_22 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAFD16SS_33_22 : AE_MULZAAFD16SS_33_22_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaafd16ss_33_22 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAFD16SS_33_22_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaafd16ss.33_22_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAFD16SS_33_22_S2 : AE_MULZAAFD16SS_33_22_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaafd16ss_33_22_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAFD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzaafd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZAAFD24_HH_LL : AE_MULZAAFD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzaafd24_hh_ll AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZAAFD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaafd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAFD24_HH_LL_S2 : AE_MULZAAFD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaafd24_hh_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAFD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzaafd24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{51} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZAAFD24_HL_LH : AE_MULZAAFD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzaafd24_hl_lh AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZAAFD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaafd24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAFD24_HL_LH_S2 : AE_MULZAAFD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaafd24_hl_lh_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAFD32X16_H0_L1_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaafd32x16.h0.l1 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAFD32X16_H0_L1 : AE_MULZAAFD32X16_H0_L1_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaafd32x16_h0_l1 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAFD32X16_H0_L1_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaafd32x16.h0.l1_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAFD32X16_H0_L1_S2 : AE_MULZAAFD32X16_H0_L1_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaafd32x16_h0_l1_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAFD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaafd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAFD32X16_H1_L0 : AE_MULZAAFD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaafd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAFD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaafd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAFD32X16_H1_L0_S2 : AE_MULZAAFD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaafd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAFD32X16_H2_L3_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaafd32x16.h2.l3 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAFD32X16_H2_L3 : AE_MULZAAFD32X16_H2_L3_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaafd32x16_h2_l3 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAFD32X16_H2_L3_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaafd32x16.h2.l3_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAFD32X16_H2_L3_S2 : AE_MULZAAFD32X16_H2_L3_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaafd32x16_h2_l3_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZAAFD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzaafd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZAAFD32X16_H3_L2 : AE_MULZAAFD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzaafd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZAAFD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzaafd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZAAFD32X16_H3_L2_S2 : AE_MULZAAFD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzaafd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZASD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzasd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZASD24_HH_LL : AE_MULZASD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzasd24_hh_ll AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZASD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzasd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZASD24_HH_LL_S2 : AE_MULZASD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzasd24_hh_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZASD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzasd24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZASD24_HL_LH : AE_MULZASD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzasd24_hl_lh AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZASD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzasd24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZASD24_HL_LH_S2 : AE_MULZASD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzasd24_hl_lh_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZASD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzasd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZASD32X16_H1_L0 : AE_MULZASD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzasd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZASD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzasd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZASD32X16_H1_L0_S2 : AE_MULZASD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzasd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZASD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzasd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{43} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZASD32X16_H3_L2 : AE_MULZASD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzasd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZASD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzasd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZASD32X16_H3_L2_S2 : AE_MULZASD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzasd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZASFD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzasfd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZASFD24_HH_LL : AE_MULZASFD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzasfd24_hh_ll AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZASFD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzasfd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZASFD24_HH_LL_S2 : AE_MULZASFD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzasfd24_hh_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZASFD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzasfd24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZASFD24_HL_LH : AE_MULZASFD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzasfd24_hl_lh AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZASFD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzasfd24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZASFD24_HL_LH_S2 : AE_MULZASFD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzasfd24_hl_lh_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZASFD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzasfd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZASFD32X16_H1_L0 : AE_MULZASFD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzasfd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZASFD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzasfd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZASFD32X16_H1_L0_S2 : AE_MULZASFD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzasfd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZASFD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzasfd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZASFD32X16_H3_L2 : AE_MULZASFD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzasfd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZASFD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzasfd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZASFD32X16_H3_L2_S2 : AE_MULZASFD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzasfd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSAD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzsad24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{50} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZSAD24_HH_LL : AE_MULZSAD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzsad24_hh_ll AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZSAD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzsad24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSAD24_HH_LL_S2 : AE_MULZSAD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzsad24_hh_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSAD32X16_H1_L0_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzsad32x16.h1.l0 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{36} = 1;
let Inst{37} = 1;
let Inst{39} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZSAD32X16_H1_L0 : AE_MULZSAD32X16_H1_L0_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzsad32x16_h1_l0 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZSAD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzsad32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSAD32X16_H1_L0_S2 : AE_MULZSAD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzsad32x16_h1_l0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSAD32X16_H3_L2_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzsad32x16.h3.l2 $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{23} = 1;
let Inst{36} = 1;
let Inst{37} = 1;
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZSAD32X16_H3_L2 : AE_MULZSAD32X16_H3_L2_AE_FORMAT2<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzsad32x16_h3_l2 AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZSAD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzsad32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSAD32X16_H3_L2_S2 : AE_MULZSAD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzsad32x16_h3_l2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSAFD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzsafd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZSAFD24_HH_LL : AE_MULZSAFD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzsafd24_hh_ll AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZSAFD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzsafd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSAFD24_HH_LL_S2 : AE_MULZSAFD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzsafd24_hh_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSAFD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzsafd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZSAFD32X16_H1_L0 : AE_MULZSAFD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzsafd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZSAFD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzsafd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSAFD32X16_H1_L0_S2 : AE_MULZSAFD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzsafd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSAFD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzsafd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZSAFD32X16_H3_L2 : AE_MULZSAFD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzsafd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZSAFD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzsafd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSAFD32X16_H3_L2_S2 : AE_MULZSAFD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzsafd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzssd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZSSD24_HH_LL : AE_MULZSSD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzssd24_hh_ll AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZSSD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSD24_HH_LL_S2 : AE_MULZSSD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssd24_hh_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzssd24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZSSD24_HL_LH : AE_MULZSSD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzssd24_hl_lh AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZSSD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssd24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSD24_HL_LH_S2 : AE_MULZSSD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssd24_hl_lh_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzssd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZSSD32X16_H1_L0 : AE_MULZSSD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzssd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZSSD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSD32X16_H1_L0_S2 : AE_MULZSSD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzssd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZSSD32X16_H3_L2 : AE_MULZSSD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzssd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZSSD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSD32X16_H3_L2_S2 : AE_MULZSSD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSFD16SS_11_00_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzssfd16ss.11_00 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZSSFD16SS_11_00 : AE_MULZSSFD16SS_11_00_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzssfd16ss_11_00 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZSSFD16SS_11_00_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssfd16ss.11_00_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSFD16SS_11_00_S2 : AE_MULZSSFD16SS_11_00_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssfd16ss_11_00_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSFD16SS_13_02_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzssfd16ss.13_02 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{40} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZSSFD16SS_13_02 : AE_MULZSSFD16SS_13_02_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzssfd16ss_13_02 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZSSFD16SS_13_02_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssfd16ss.13_02_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSFD16SS_13_02_S2 : AE_MULZSSFD16SS_13_02_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssfd16ss_13_02_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSFD16SS_33_22_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzssfd16ss.33_22 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZSSFD16SS_33_22 : AE_MULZSSFD16SS_33_22_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzssfd16ss_33_22 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZSSFD16SS_33_22_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssfd16ss.33_22_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSFD16SS_33_22_S2 : AE_MULZSSFD16SS_33_22_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssfd16ss_33_22_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSFD24_HH_LL_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzssfd24.hh.ll $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZSSFD24_HH_LL : AE_MULZSSFD24_HH_LL_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzssfd24_hh_ll AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZSSFD24_HH_LL_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssfd24.hh.ll_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSFD24_HH_LL_S2 : AE_MULZSSFD24_HH_LL_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssfd24_hh_ll_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSFD24_HL_LH_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_q0), (ins AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1), "ae_mulzssfd24.hl.lh $ae_mul_q0, $ae_mul_d0, $ae_mul_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_q0;
bits<4> ae_mul_d0;
bits<4> ae_mul_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_mul_q0{0};
let Inst{17} = ae_mul_q0{1};
let Inst{18} = ae_mul_q0{2};
let Inst{19} = ae_mul_q0{3};
let Inst{24} = ae_mul_d0{0};
let Inst{25} = ae_mul_d0{1};
let Inst{26} = ae_mul_d0{2};
let Inst{27} = ae_mul_d0{3};
let Inst{32} = ae_mul_d1{0};
let Inst{33} = ae_mul_d1{1};
let Inst{34} = ae_mul_d1{2};
let Inst{35} = ae_mul_d1{3};
}



def AE_MULZSSFD24_HL_LH : AE_MULZSSFD24_HL_LH_AE_FORMAT<[(set AE_DR:$ae_mul_q0, (int_xtensa_ae_mulzssfd24_hl_lh AE_DR:$ae_mul_d0, AE_DR:$ae_mul_d1))]>;

class AE_MULZSSFD24_HL_LH_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssfd24.hl.lh_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSFD24_HL_LH_S2 : AE_MULZSSFD24_HL_LH_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssfd24_hl_lh_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSFD32X16_H1_L0_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzssfd32x16.h1.l0 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{39} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZSSFD32X16_H1_L0 : AE_MULZSSFD32X16_H1_L0_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzssfd32x16_h1_l0 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZSSFD32X16_H1_L0_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssfd32x16.h1.l0_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSFD32X16_H1_L0_S2 : AE_MULZSSFD32X16_H1_L0_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssfd32x16_h1_l0_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_MULZSSFD32X16_H3_L2_AE_FORMAT48_3<list<dag> pattern>
    : XtensaInst48<(outs AE_DR:$opnd_ae_sem_mul_x2_S1_q0), (ins AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1), "ae_mulzssfd32x16.h3.l2 $opnd_ae_sem_mul_x2_S1_q0, $opnd_ae_sem_mul_x2_S1_d0, $opnd_ae_sem_mul_x2_S1_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> opnd_ae_sem_mul_x2_S1_q0;
bits<4> opnd_ae_sem_mul_x2_S1_d0;
bits<4> opnd_ae_sem_mul_x2_S1_d1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{42} = 1;
let Inst{44} = 1;
let Inst{45} = 1;
let Inst{46} = 1;
//operands
let Inst{16} = opnd_ae_sem_mul_x2_S1_q0{0};
let Inst{17} = opnd_ae_sem_mul_x2_S1_q0{1};
let Inst{18} = opnd_ae_sem_mul_x2_S1_q0{2};
let Inst{19} = opnd_ae_sem_mul_x2_S1_q0{3};
let Inst{20} = opnd_ae_sem_mul_x2_S1_d0{0};
let Inst{21} = opnd_ae_sem_mul_x2_S1_d0{1};
let Inst{22} = opnd_ae_sem_mul_x2_S1_d0{2};
let Inst{23} = opnd_ae_sem_mul_x2_S1_d0{3};
let Inst{24} = opnd_ae_sem_mul_x2_S1_d1{0};
let Inst{25} = opnd_ae_sem_mul_x2_S1_d1{1};
let Inst{26} = opnd_ae_sem_mul_x2_S1_d1{2};
let Inst{27} = opnd_ae_sem_mul_x2_S1_d1{3};
}



def AE_MULZSSFD32X16_H3_L2 : AE_MULZSSFD32X16_H3_L2_AE_FORMAT48_3<[(set AE_DR:$opnd_ae_sem_mul_x2_S1_q0, (int_xtensa_ae_mulzssfd32x16_h3_l2 AE_DR:$opnd_ae_sem_mul_x2_S1_d0, AE_DR:$opnd_ae_sem_mul_x2_S1_d1))]>;

class AE_MULZSSFD32X16_H3_L2_S2_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_mul_S2_q0), (ins AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1), "ae_mulzssfd32x16.h3.l2_s2 $ae_mul_S2_q0, $ae_mul_S2_d0, $ae_mul_S2_d1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_mul_S2_q0;
bits<4> ae_mul_S2_d0;
bits<4> ae_mul_S2_d1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_mul_S2_q0{0};
let Inst{21} = ae_mul_S2_q0{1};
let Inst{22} = ae_mul_S2_q0{2};
let Inst{23} = ae_mul_S2_q0{3};
let Inst{36} = ae_mul_S2_d0{0};
let Inst{37} = ae_mul_S2_d0{1};
let Inst{38} = ae_mul_S2_d0{2};
let Inst{39} = ae_mul_S2_d0{3};
let Inst{28} = ae_mul_S2_d1{0};
let Inst{29} = ae_mul_S2_d1{1};
let Inst{30} = ae_mul_S2_d1{2};
let Inst{31} = ae_mul_S2_d1{3};
}



def AE_MULZSSFD32X16_H3_L2_S2 : AE_MULZSSFD32X16_H3_L2_S2_AE_FORMAT<[(set AE_DR:$ae_mul_S2_q0, (int_xtensa_ae_mulzssfd32x16_h3_l2_s2 AE_DR:$ae_mul_S2_d0, AE_DR:$ae_mul_S2_d1))]>;

class AE_NAND_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_dr_to_dr_v), (ins AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1), "ae_nand $ae_dr_to_dr_v, $ae_dr_to_dr_v0, $ae_dr_to_dr_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_dr_to_dr_v;
bits<4> ae_dr_to_dr_v0;
bits<4> ae_dr_to_dr_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_dr_to_dr_v{0};
let Inst{21} = ae_dr_to_dr_v{1};
let Inst{22} = ae_dr_to_dr_v{2};
let Inst{23} = ae_dr_to_dr_v{3};
let Inst{36} = ae_dr_to_dr_v0{0};
let Inst{37} = ae_dr_to_dr_v0{1};
let Inst{38} = ae_dr_to_dr_v0{2};
let Inst{39} = ae_dr_to_dr_v0{3};
let Inst{28} = ae_dr_to_dr_v1{0};
let Inst{29} = ae_dr_to_dr_v1{1};
let Inst{30} = ae_dr_to_dr_v1{2};
let Inst{31} = ae_dr_to_dr_v1{3};
}



def AE_NAND : AE_NAND_AE_FORMAT1<[(set AE_DR:$ae_dr_to_dr_v, (int_xtensa_ae_nand AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1))]>;

class AE_NEG16S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_neg16s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_NEG16S : AE_NEG16S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_neg16s AE_DR:$ae_arth_v1))]>;

class AE_NEG24S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_neg24s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{27} = 1;
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_NEG24S : AE_NEG24S_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_neg24s AE_DR:$ae_arth_v1))]>;

class AE_NEG32_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_neg32 $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{24} = 1;
let Inst{27} = 1;
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_NEG32 : AE_NEG32_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_neg32 AE_DR:$ae_arth_v1))]>;

class AE_NEG32S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_neg32s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_NEG32S : AE_NEG32S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_neg32s AE_DR:$ae_arth_v1))]>;

class AE_NEG64_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_neg64 $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{10} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_NEG64 : AE_NEG64_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_neg64 AE_DR:$ae_arth_v1))]>;

class AE_NEG64S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_neg64s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{26} = 1;
let Inst{27} = 1;
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{51} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_NEG64S : AE_NEG64S_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_neg64s AE_DR:$ae_arth_v1))]>;

class AE_NSA64_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_nsa64 $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ae_dr_to_ar_v0{0};
let Inst{9} = ae_dr_to_ar_v0{1};
let Inst{10} = ae_dr_to_ar_v0{2};
let Inst{11} = ae_dr_to_ar_v0{3};
}



def AE_NSA64 : AE_NSA64_X24<[(set AR:$arr, (int_xtensa_ae_nsa64 AE_DR:$ae_dr_to_ar_v0))]>;

class AE_NSAZ16_0_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_nsaz16.0 $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ae_dr_to_ar_v0{0};
let Inst{9} = ae_dr_to_ar_v0{1};
let Inst{10} = ae_dr_to_ar_v0{2};
let Inst{11} = ae_dr_to_ar_v0{3};
}



def AE_NSAZ16_0 : AE_NSAZ16_0_X24<[(set AR:$arr, (int_xtensa_ae_nsaz16_0 AE_DR:$ae_dr_to_ar_v0))]>;

class AE_NSAZ32_L_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AE_DR:$ae_dr_to_ar_v0), "ae_nsaz32.l $arr, $ae_dr_to_ar_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ae_dr_to_ar_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ae_dr_to_ar_v0{0};
let Inst{9} = ae_dr_to_ar_v0{1};
let Inst{10} = ae_dr_to_ar_v0{2};
let Inst{11} = ae_dr_to_ar_v0{3};
}



def AE_NSAZ32_L : AE_NSAZ32_L_X24<[(set AR:$arr, (int_xtensa_ae_nsaz32_l AE_DR:$ae_dr_to_ar_v0))]>;

class AE_OR_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_dr_to_dr_v), (ins AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1), "ae_or $ae_dr_to_dr_v, $ae_dr_to_dr_v0, $ae_dr_to_dr_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_dr_to_dr_v;
bits<4> ae_dr_to_dr_v0;
bits<4> ae_dr_to_dr_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
//operands
let Inst{12} = ae_dr_to_dr_v{0};
let Inst{13} = ae_dr_to_dr_v{1};
let Inst{14} = ae_dr_to_dr_v{2};
let Inst{15} = ae_dr_to_dr_v{3};
let Inst{8} = ae_dr_to_dr_v0{0};
let Inst{9} = ae_dr_to_dr_v0{1};
let Inst{10} = ae_dr_to_dr_v0{2};
let Inst{11} = ae_dr_to_dr_v0{3};
let Inst{4} = ae_dr_to_dr_v1{0};
let Inst{5} = ae_dr_to_dr_v1{1};
let Inst{6} = ae_dr_to_dr_v1{2};
let Inst{7} = ae_dr_to_dr_v1{3};
}



def AE_OR : AE_OR_X24<[(set AE_DR:$ae_dr_to_dr_v, (int_xtensa_ae_or AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1))]>;

class AE_PKSR24_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_pks_d_out), (ins AE_DR:$ae_pks_d, AE_DR:$ae_pks_s, uimm2:$ae_imm2), "ae_pksr24 $ae_pks_d, $ae_pks_s, $ae_imm2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_pks_d;
bits<4> ae_pks_s;
bits<2> ae_imm2;
let Constraints = "$ae_pks_d = $ae_pks_d_out,@earlyclobber $ae_pks_d_out";


//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_pks_d{0};
let Inst{13} = ae_pks_d{1};
let Inst{14} = ae_pks_d{2};
let Inst{15} = ae_pks_d{3};
let Inst{4} = ae_pks_s{0};
let Inst{5} = ae_pks_s{1};
let Inst{6} = ae_pks_s{2};
let Inst{7} = ae_pks_s{3};
let Inst{10} = ae_imm2{0};
let Inst{11} = ae_imm2{1};
}



def AE_PKSR24 : AE_PKSR24_X24<[(set AE_DR:$ae_pks_d_out, (int_xtensa_ae_pksr24 AE_DR:$ae_pks_d, AE_DR:$ae_pks_s, timm:$ae_imm2))]>;

class AE_PKSR32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_pks_d_out), (ins AE_DR:$ae_pks_d, AE_DR:$ae_pks_s, uimm2:$ae_imm2), "ae_pksr32 $ae_pks_d, $ae_pks_s, $ae_imm2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_pks_d;
bits<4> ae_pks_s;
bits<2> ae_imm2;
let Constraints = "$ae_pks_d = $ae_pks_d_out,@earlyclobber $ae_pks_d_out";


//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_pks_d{0};
let Inst{13} = ae_pks_d{1};
let Inst{14} = ae_pks_d{2};
let Inst{15} = ae_pks_d{3};
let Inst{4} = ae_pks_s{0};
let Inst{5} = ae_pks_s{1};
let Inst{6} = ae_pks_s{2};
let Inst{7} = ae_pks_s{3};
let Inst{10} = ae_imm2{0};
let Inst{11} = ae_imm2{1};
}



def AE_PKSR32 : AE_PKSR32_X24<[(set AE_DR:$ae_pks_d_out, (int_xtensa_ae_pksr32 AE_DR:$ae_pks_d, AE_DR:$ae_pks_s, timm:$ae_imm2))]>;

class AE_ROUND16X4F32SASYM_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1, AE_DR:$ae_arth_v0), "ae_round16x4f32sasym $ae_arth_v, $ae_arth_v1, $ae_arth_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;
bits<4> ae_arth_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
}



def AE_ROUND16X4F32SASYM : AE_ROUND16X4F32SASYM_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_round16x4f32sasym AE_DR:$ae_arth_v1, AE_DR:$ae_arth_v0))]>;

class AE_ROUND16X4F32SSYM_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1, AE_DR:$ae_arth_v0), "ae_round16x4f32ssym $ae_arth_v, $ae_arth_v1, $ae_arth_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;
bits<4> ae_arth_v0;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{36} = ae_arth_v1{0};
let Inst{37} = ae_arth_v1{1};
let Inst{38} = ae_arth_v1{2};
let Inst{39} = ae_arth_v1{3};
let Inst{28} = ae_arth_v0{0};
let Inst{29} = ae_arth_v0{1};
let Inst{30} = ae_arth_v0{2};
let Inst{31} = ae_arth_v0{3};
}



def AE_ROUND16X4F32SSYM : AE_ROUND16X4F32SSYM_AE_FORMAT1<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_round16x4f32ssym AE_DR:$ae_arth_v1, AE_DR:$ae_arth_v0))]>;

class AE_ROUND24X2F48SASYM_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_round24x2f48sasym $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ROUND24X2F48SASYM : AE_ROUND24X2F48SASYM_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_round24x2f48sasym AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ROUND24X2F48SSYM_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_round24x2f48ssym $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ROUND24X2F48SSYM : AE_ROUND24X2F48SSYM_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_round24x2f48ssym AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ROUND32X2F48SASYM_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_round32x2f48sasym $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{59} = 1;
let Inst{60} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{28} = ae_arth_v0{0};
let Inst{29} = ae_arth_v0{1};
let Inst{30} = ae_arth_v0{2};
let Inst{31} = ae_arth_v0{3};
let Inst{36} = ae_arth_v1{0};
let Inst{37} = ae_arth_v1{1};
let Inst{38} = ae_arth_v1{2};
let Inst{39} = ae_arth_v1{3};
}



def AE_ROUND32X2F48SASYM : AE_ROUND32X2F48SASYM_AE_FORMAT1<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_round32x2f48sasym AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ROUND32X2F48SSYM_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_round32x2f48ssym $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{61} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{28} = ae_arth_v0{0};
let Inst{29} = ae_arth_v0{1};
let Inst{30} = ae_arth_v0{2};
let Inst{31} = ae_arth_v0{3};
let Inst{36} = ae_arth_v1{0};
let Inst{37} = ae_arth_v1{1};
let Inst{38} = ae_arth_v1{2};
let Inst{39} = ae_arth_v1{3};
}



def AE_ROUND32X2F48SSYM : AE_ROUND32X2F48SSYM_AE_FORMAT1<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_round32x2f48ssym AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ROUND32X2F64SASYM_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_round32x2f64sasym $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ROUND32X2F64SASYM : AE_ROUND32X2F64SASYM_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_round32x2f64sasym AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ROUND32X2F64SSYM_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_round32x2f64ssym $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{57} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{28} = ae_arth_v0{0};
let Inst{29} = ae_arth_v0{1};
let Inst{30} = ae_arth_v0{2};
let Inst{31} = ae_arth_v0{3};
let Inst{36} = ae_arth_v1{0};
let Inst{37} = ae_arth_v1{1};
let Inst{38} = ae_arth_v1{2};
let Inst{39} = ae_arth_v1{3};
}



def AE_ROUND32X2F64SSYM : AE_ROUND32X2F64SSYM_AE_FORMAT1<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_round32x2f64ssym AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ROUNDSP16F24ASYM_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0), "ae_roundsp16f24asym $ae_arth_v, $ae_arth_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{28} = ae_arth_v0{0};
let Inst{29} = ae_arth_v0{1};
let Inst{30} = ae_arth_v0{2};
let Inst{31} = ae_arth_v0{3};
}



def AE_ROUNDSP16F24ASYM : AE_ROUNDSP16F24ASYM_AE_FORMAT1<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_roundsp16f24asym AE_DR:$ae_arth_v0))]>;

class AE_ROUNDSP16F24SYM_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0), "ae_roundsp16f24sym $ae_arth_v, $ae_arth_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{36} = 1;
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{28} = ae_arth_v0{0};
let Inst{29} = ae_arth_v0{1};
let Inst{30} = ae_arth_v0{2};
let Inst{31} = ae_arth_v0{3};
}



def AE_ROUNDSP16F24SYM : AE_ROUNDSP16F24SYM_AE_FORMAT1<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_roundsp16f24sym AE_DR:$ae_arth_v0))]>;

class AE_ROUNDSP16Q48X2ASYM_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_roundsp16q48x2asym $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ROUNDSP16Q48X2ASYM : AE_ROUNDSP16Q48X2ASYM_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_roundsp16q48x2asym AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ROUNDSP16Q48X2SYM_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_roundsp16q48x2sym $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{58} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{28} = ae_arth_v0{0};
let Inst{29} = ae_arth_v0{1};
let Inst{30} = ae_arth_v0{2};
let Inst{31} = ae_arth_v0{3};
let Inst{36} = ae_arth_v1{0};
let Inst{37} = ae_arth_v1{1};
let Inst{38} = ae_arth_v1{2};
let Inst{39} = ae_arth_v1{3};
}



def AE_ROUNDSP16Q48X2SYM : AE_ROUNDSP16Q48X2SYM_AE_FORMAT1<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_roundsp16q48x2sym AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_ROUNDSQ32F48ASYM_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_roundsq32f48asym $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{11} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ROUNDSQ32F48ASYM : AE_ROUNDSQ32F48ASYM_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_roundsq32f48asym AE_DR:$ae_arth_v1))]>;

class AE_ROUNDSQ32F48SYM_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_roundsq32f48sym $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{11} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_ROUNDSQ32F48SYM : AE_ROUNDSQ32F48SYM_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_roundsq32f48sym AE_DR:$ae_arth_v1))]>;

class AE_S16_0_I_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, imm16n_14:$ae_immls16), "ae_s16.0.i $ae_ls_v, $ars, $ae_immls16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls16;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls16{0};
let Inst{5} = ae_immls16{1};
let Inst{6} = ae_immls16{2};
let Inst{7} = ae_immls16{3};
}



def AE_S16_0_I : AE_S16_0_I_X24<[(int_xtensa_ae_s16_0_i AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls16)]>;

class AE_S16_0_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, imm16n_14:$ae_immls16), "ae_s16.0.ip $ae_ls_v, $ars, $ae_immls16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls16;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls16{0};
let Inst{5} = ae_immls16{1};
let Inst{6} = ae_immls16{2};
let Inst{7} = ae_immls16{3};
}



def AE_S16_0_IP : AE_S16_0_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_s16_0_ip AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls16))]>;

class AE_S16_0_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16.0.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16_0_X : AE_S16_0_X_AE_FORMAT48<[(int_xtensa_ae_s16_0_x AE_DR:$ae_ls_v, AR:$ars, AR:$art)]>;

class AE_S16_0_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16.0.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16_0_XC : AE_S16_0_XC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s16_0_xc AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S16_0_XP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16.0.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16_0_XP : AE_S16_0_XP_X24<[(set AR:$ars_out, (int_xtensa_ae_s16_0_xp AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S16M_L_I_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, imm16n_14:$ae_immls16), "ae_s16m.l.i $ae_ls_v, $ars, $ae_immls16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls16;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls16{0};
let Inst{5} = ae_immls16{1};
let Inst{6} = ae_immls16{2};
let Inst{7} = ae_immls16{3};
}



def AE_S16M_L_I : AE_S16M_L_I_X24<[(int_xtensa_ae_s16m_l_i AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls16)]>;

class AE_S16M_L_IU_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, imm16n_14:$ae_immls16), "ae_s16m.l.iu $ae_ls_v, $ars, $ae_immls16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls16;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls16{0};
let Inst{5} = ae_immls16{1};
let Inst{6} = ae_immls16{2};
let Inst{7} = ae_immls16{3};
}



def AE_S16M_L_IU : AE_S16M_L_IU_X24<[(set AR:$ars_out, (int_xtensa_ae_s16m_l_iu AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls16))]>;

class AE_S16M_L_X_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16m.l.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16M_L_X : AE_S16M_L_X_X24<[(int_xtensa_ae_s16m_l_x AE_DR:$ae_ls_v, AR:$ars, AR:$art)]>;

class AE_S16M_L_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16m.l.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16M_L_XC : AE_S16M_L_XC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s16m_l_xc AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S16M_L_XU_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16m.l.xu $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16M_L_XU : AE_S16M_L_XU_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s16m_l_xu AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S16X2M_I_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, imm32n_28:$ae_immls32), "ae_s16x2m.i $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S16X2M_I : AE_S16X2M_I_X24<[(int_xtensa_ae_s16x2m_i AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls32)]>;

class AE_S16X2M_IU_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, imm32n_28:$ae_immls32), "ae_s16x2m.iu $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S16X2M_IU : AE_S16X2M_IU_X24<[(set AR:$ars_out, (int_xtensa_ae_s16x2m_iu AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls32))]>;

class AE_S16X2M_X_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16x2m.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16X2M_X : AE_S16X2M_X_X24<[(int_xtensa_ae_s16x2m_x AE_DR:$ae_ls_v, AR:$ars, AR:$art)]>;

class AE_S16X2M_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16x2m.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16X2M_XC : AE_S16X2M_XC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s16x2m_xc AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S16X2M_XU_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16x2m.xu $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16X2M_XU : AE_S16X2M_XU_X24<[(set AR:$ars_out, (int_xtensa_ae_s16x2m_xu AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S16X4_I_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, imm64n_56:$ae_immls64), "ae_s16x4.i $ae_ls_v, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls64;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{6} = ae_immls64{2};
let Inst{7} = ae_immls64{3};
}



def AE_S16X4_I : AE_S16X4_I_X24<[(int_xtensa_ae_s16x4_i AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls64)]>;

class AE_S16X4_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, imm0_56:$ae_immls64pos), "ae_s16x4.ip $ae_ls_v, $ars, $ae_immls64pos", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<3> ae_immls64pos;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64pos{0};
let Inst{5} = ae_immls64pos{1};
let Inst{6} = ae_immls64pos{2};
}



def AE_S16X4_IP : AE_S16X4_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_s16x4_ip AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls64pos))]>;

class AE_S16X4_RIC_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars), "ae_s16x4.ric $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{4} = 1;
let Inst{6} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_S16X4_RIC : AE_S16X4_RIC_AE_FORMAT<[(set AR:$ars_out, (int_xtensa_ae_s16x4_ric AE_DR:$ae_ls_v, AR:$ars))]>;

class AE_S16X4_RIP_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars), "ae_s16x4.rip $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_S16X4_RIP : AE_S16X4_RIP_AE_FORMAT<[(set AR:$ars_out, (int_xtensa_ae_s16x4_rip AE_DR:$ae_ls_v, AR:$ars))]>;

class AE_S16X4_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16x4.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{35} = 1;
let Inst{36} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16X4_X : AE_S16X4_X_AE_FORMAT48<[(int_xtensa_ae_s16x4_x AE_DR:$ae_ls_v, AR:$ars, AR:$art)]>;

class AE_S16X4_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16x4.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16X4_XC : AE_S16X4_XC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s16x4_xc AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S16X4_XP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s16x4.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S16X4_XP : AE_S16X4_XP_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s16x4_xp AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S24RA64S_I_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs ), (ins AE_DR:$ae_ls_v1, AR:$ars, imm32n_28:$ae_immls32), "ae_s24ra64s.i $ae_ls_v1, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v1;
bits<4> ars;
bits<4> ae_immls32;


let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S24RA64S_I : AE_S24RA64S_I_AE_FORMAT48<[(int_xtensa_ae_s24ra64s_i AE_DR:$ae_ls_v1, AR:$ars, timm:$ae_immls32)]>;

class AE_S24RA64S_IP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v1, AR:$ars, imm32n_28:$ae_immls32), "ae_s24ra64s.ip $ae_ls_v1, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v1;
bits<4> ars;
bits<4> ae_immls32;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S24RA64S_IP : AE_S24RA64S_IP_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s24ra64s_ip AE_DR:$ae_ls_v1, AR:$ars, timm:$ae_immls32))]>;

class AE_S24RA64S_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs ), (ins AE_DR:$ae_ls_v1, AR:$ars, AR:$art), "ae_s24ra64s.x $ae_ls_v1, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v1;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{30} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S24RA64S_X : AE_S24RA64S_X_AE_FORMAT48<[(int_xtensa_ae_s24ra64s_x AE_DR:$ae_ls_v1, AR:$ars, AR:$art)]>;

class AE_S24RA64S_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v1, AR:$ars, AR:$art), "ae_s24ra64s.xc $ae_ls_v1, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v1;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S24RA64S_XC : AE_S24RA64S_XC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s24ra64s_xc AE_DR:$ae_ls_v1, AR:$ars, AR:$art))]>;

class AE_S24RA64S_XP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v1, AR:$ars, AR:$art), "ae_s24ra64s.xp $ae_ls_v1, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v1;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S24RA64S_XP : AE_S24RA64S_XP_X24<[(set AR:$ars_out, (int_xtensa_ae_s24ra64s_xp AE_DR:$ae_ls_v1, AR:$ars, AR:$art))]>;

class AE_S24X2RA64S_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v2, AE_DR:$ae_ls_v1, AR:$ars), "ae_s24x2ra64s.ip $ae_ls_v2, $ae_ls_v1, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v2;
bits<4> ae_ls_v1;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
//operands
let Inst{4} = ae_ls_v2{0};
let Inst{5} = ae_ls_v2{1};
let Inst{6} = ae_ls_v2{2};
let Inst{7} = ae_ls_v2{3};
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_S24X2RA64S_IP : AE_S24X2RA64S_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_s24x2ra64s_ip AE_DR:$ae_ls_v2, AE_DR:$ae_ls_v1, AR:$ars))]>;

class AE_S32_L_I_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, imm32n_28:$ae_immls32), "ae_s32.l.i $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S32_L_I : AE_S32_L_I_X24<[(int_xtensa_ae_s32_l_i AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls32)]>;

class AE_S32_L_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, imm32n_28:$ae_immls32), "ae_s32.l.ip $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S32_L_IP : AE_S32_L_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_s32_l_ip AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls32))]>;

class AE_S32_L_X_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32.l.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32_L_X : AE_S32_L_X_X24<[(int_xtensa_ae_s32_l_x AE_DR:$ae_ls_v, AR:$ars, AR:$art)]>;

class AE_S32_L_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32.l.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32_L_XC : AE_S32_L_XC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s32_l_xc AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S32_L_XP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32.l.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32_L_XP : AE_S32_L_XP_X24<[(set AR:$ars_out, (int_xtensa_ae_s32_l_xp AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S32F24_L_I_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, imm32n_28:$ae_immls32), "ae_s32f24.l.i $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S32F24_L_I : AE_S32F24_L_I_X24<[(int_xtensa_ae_s32f24_l_i AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls32)]>;

class AE_S32F24_L_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, imm32n_28:$ae_immls32), "ae_s32f24.l.ip $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S32F24_L_IP : AE_S32F24_L_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_s32f24_l_ip AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls32))]>;

class AE_S32F24_L_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32f24.l.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{31} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32F24_L_X : AE_S32F24_L_X_AE_FORMAT48<[(int_xtensa_ae_s32f24_l_x AE_DR:$ae_ls_v, AR:$ars, AR:$art)]>;

class AE_S32F24_L_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32f24.l.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{31} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32F24_L_XC : AE_S32F24_L_XC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s32f24_l_xc AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S32F24_L_XP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32f24.l.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{31} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32F24_L_XP : AE_S32F24_L_XP_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s32f24_l_xp AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S32M_I_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, imm32n_28:$ae_immls32), "ae_s32m.i $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S32M_I : AE_S32M_I_X24<[(int_xtensa_ae_s32m_i AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls32)]>;

class AE_S32M_IU_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, imm32n_28:$ae_immls32), "ae_s32m.iu $ae_ls_v, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls32;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S32M_IU : AE_S32M_IU_X24<[(set AR:$ars_out, (int_xtensa_ae_s32m_iu AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls32))]>;

class AE_S32M_X_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32m.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32M_X : AE_S32M_X_X24<[(int_xtensa_ae_s32m_x AE_DR:$ae_ls_v, AR:$ars, AR:$art)]>;

class AE_S32M_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32m.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32M_XC : AE_S32M_XC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s32m_xc AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S32M_XU_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32m.xu $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32M_XU : AE_S32M_XU_X24<[(set AR:$ars_out, (int_xtensa_ae_s32m_xu AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S32RA64S_I_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs ), (ins AE_DR:$ae_ls_v1, AR:$ars, imm32n_28:$ae_immls32), "ae_s32ra64s.i $ae_ls_v1, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v1;
bits<4> ars;
bits<4> ae_immls32;


let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S32RA64S_I : AE_S32RA64S_I_AE_FORMAT48<[(int_xtensa_ae_s32ra64s_i AE_DR:$ae_ls_v1, AR:$ars, timm:$ae_immls32)]>;

class AE_S32RA64S_IP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v1, AR:$ars, imm32n_28:$ae_immls32), "ae_s32ra64s.ip $ae_ls_v1, $ars, $ae_immls32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v1;
bits<4> ars;
bits<4> ae_immls32;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{33} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls32{0};
let Inst{5} = ae_immls32{1};
let Inst{6} = ae_immls32{2};
let Inst{7} = ae_immls32{3};
}



def AE_S32RA64S_IP : AE_S32RA64S_IP_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s32ra64s_ip AE_DR:$ae_ls_v1, AR:$ars, timm:$ae_immls32))]>;

class AE_S32RA64S_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs ), (ins AE_DR:$ae_ls_v1, AR:$ars, AR:$art), "ae_s32ra64s.x $ae_ls_v1, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v1;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32RA64S_X : AE_S32RA64S_X_AE_FORMAT48<[(int_xtensa_ae_s32ra64s_x AE_DR:$ae_ls_v1, AR:$ars, AR:$art)]>;

class AE_S32RA64S_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v1, AR:$ars, AR:$art), "ae_s32ra64s.xc $ae_ls_v1, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v1;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32RA64S_XC : AE_S32RA64S_XC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s32ra64s_xc AE_DR:$ae_ls_v1, AR:$ars, AR:$art))]>;

class AE_S32RA64S_XP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v1, AR:$ars, AR:$art), "ae_s32ra64s.xp $ae_ls_v1, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v1;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
//operands
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32RA64S_XP : AE_S32RA64S_XP_X24<[(set AR:$ars_out, (int_xtensa_ae_s32ra64s_xp AE_DR:$ae_ls_v1, AR:$ars, AR:$art))]>;

class AE_S32X2_I_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, imm64n_56:$ae_immls64), "ae_s32x2.i $ae_ls_v, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls64;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{6} = ae_immls64{2};
let Inst{7} = ae_immls64{3};
}



def AE_S32X2_I : AE_S32X2_I_X24<[(int_xtensa_ae_s32x2_i AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls64)]>;

class AE_S32X2_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, imm0_56:$ae_immls64pos), "ae_s32x2.ip $ae_ls_v, $ars, $ae_immls64pos", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<3> ae_immls64pos;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64pos{0};
let Inst{5} = ae_immls64pos{1};
let Inst{6} = ae_immls64pos{2};
}



def AE_S32X2_IP : AE_S32X2_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_s32x2_ip AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls64pos))]>;

class AE_S32X2_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars), "ae_s32x2.ric $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_S32X2_RIC : AE_S32X2_RIC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s32x2_ric AE_DR:$ae_ls_v, AR:$ars))]>;

class AE_S32X2_RIP_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars), "ae_s32x2.rip $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_S32X2_RIP : AE_S32X2_RIP_AE_FORMAT<[(set AR:$ars_out, (int_xtensa_ae_s32x2_rip AE_DR:$ae_ls_v, AR:$ars))]>;

class AE_S32X2_X_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32x2.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32X2_X : AE_S32X2_X_X24<[(int_xtensa_ae_s32x2_x AE_DR:$ae_ls_v, AR:$ars, AR:$art)]>;

class AE_S32X2_XC_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32x2.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32X2_XC : AE_S32X2_XC_X24<[(set AR:$ars_out, (int_xtensa_ae_s32x2_xc AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S32X2_XP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32x2.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32X2_XP : AE_S32X2_XP_X24<[(set AR:$ars_out, (int_xtensa_ae_s32x2_xp AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S32X2F24_I_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, imm64n_56:$ae_immls64), "ae_s32x2f24.i $ae_ls_v, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls64;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{6} = ae_immls64{2};
let Inst{7} = ae_immls64{3};
}



def AE_S32X2F24_I : AE_S32X2F24_I_X24<[(int_xtensa_ae_s32x2f24_i AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls64)]>;

class AE_S32X2F24_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, imm0_56:$ae_immls64pos), "ae_s32x2f24.ip $ae_ls_v, $ars, $ae_immls64pos", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<3> ae_immls64pos;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64pos{0};
let Inst{5} = ae_immls64pos{1};
let Inst{6} = ae_immls64pos{2};
}



def AE_S32X2F24_IP : AE_S32X2F24_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_s32x2f24_ip AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls64pos))]>;

class AE_S32X2F24_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars), "ae_s32x2f24.ric $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{6} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_S32X2F24_RIC : AE_S32X2F24_RIC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s32x2f24_ric AE_DR:$ae_ls_v, AR:$ars))]>;

class AE_S32X2F24_RIP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars), "ae_s32x2f24.rip $ae_ls_v, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{7} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_S32X2F24_RIP : AE_S32X2F24_RIP_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s32x2f24_rip AE_DR:$ae_ls_v, AR:$ars))]>;

class AE_S32X2F24_X_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32x2f24.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32X2F24_X : AE_S32X2F24_X_X24<[(int_xtensa_ae_s32x2f24_x AE_DR:$ae_ls_v, AR:$ars, AR:$art)]>;

class AE_S32X2F24_XC_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32x2f24.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32X2F24_XC : AE_S32X2F24_XC_X24<[(set AR:$ars_out, (int_xtensa_ae_s32x2f24_xc AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S32X2F24_XP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s32x2f24.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S32X2F24_XP : AE_S32X2F24_XP_X24<[(set AR:$ars_out, (int_xtensa_ae_s32x2f24_xp AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S32X2RA64S_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v2, AE_DR:$ae_ls_v1, AR:$ars), "ae_s32x2ra64s.ip $ae_ls_v2, $ae_ls_v1, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v2;
bits<4> ae_ls_v1;
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
//operands
let Inst{4} = ae_ls_v2{0};
let Inst{5} = ae_ls_v2{1};
let Inst{6} = ae_ls_v2{2};
let Inst{7} = ae_ls_v2{3};
let Inst{12} = ae_ls_v1{0};
let Inst{13} = ae_ls_v1{1};
let Inst{14} = ae_ls_v1{2};
let Inst{15} = ae_ls_v1{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_S32X2RA64S_IP : AE_S32X2RA64S_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_s32x2ra64s_ip AE_DR:$ae_ls_v2, AE_DR:$ae_ls_v1, AR:$ars))]>;

class AE_S64_I_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, imm64n_56:$ae_immls64), "ae_s64.i $ae_ls_v, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls64;


let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{6} = ae_immls64{2};
let Inst{7} = ae_immls64{3};
}



def AE_S64_I : AE_S64_I_X24<[(int_xtensa_ae_s64_i AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls64)]>;

class AE_S64_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, imm64n_56:$ae_immls64), "ae_s64.ip $ae_ls_v, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> ae_immls64;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{6} = ae_immls64{2};
let Inst{7} = ae_immls64{3};
}



def AE_S64_IP : AE_S64_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_s64_ip AE_DR:$ae_ls_v, AR:$ars, timm:$ae_immls64))]>;

class AE_S64_X_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs ), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s64.x $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;


let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S64_X : AE_S64_X_AE_FORMAT48<[(int_xtensa_ae_s64_x AE_DR:$ae_ls_v, AR:$ars, AR:$art)]>;

class AE_S64_XC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s64.xc $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{28} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S64_XC : AE_S64_XC_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s64_xc AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_S64_XP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AR:$ars_out), (ins AE_DR:$ae_ls_v, AR:$ars, AR:$art), "ae_s64.xp $ae_ls_v, $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_S64_XP : AE_S64_XP_AE_FORMAT48<[(set AR:$ars_out, (int_xtensa_ae_s64_xp AE_DR:$ae_ls_v, AR:$ars, AR:$art))]>;

class AE_SA16X4_IC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa16x4.ic $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA16X4_IC : AE_SA16X4_IC_AE_FORMAT48<[]>;

class AE_SA16X4_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa16x4.ip $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{4} = ae_ls_su{0};
let Inst{5} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA16X4_IP : AE_SA16X4_IP_X24<[]>;

class AE_SA16X4_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa16x4.ric $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA16X4_RIC : AE_SA16X4_RIC_AE_FORMAT48<[]>;

class AE_SA16X4_RIP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa16x4.rip $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{29} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA16X4_RIP : AE_SA16X4_RIP_AE_FORMAT48<[]>;

class AE_SA24_L_IC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa24.l.ic $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA24_L_IC : AE_SA24_L_IC_AE_FORMAT48<[]>;

class AE_SA24_L_IP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa24.l.ip $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA24_L_IP : AE_SA24_L_IP_AE_FORMAT48<[]>;

class AE_SA24_L_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa24.l.ric $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA24_L_RIC : AE_SA24_L_RIC_AE_FORMAT48<[]>;

class AE_SA24_L_RIP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa24.l.rip $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA24_L_RIP : AE_SA24_L_RIP_AE_FORMAT48<[]>;

class AE_SA24X2_IC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa24x2.ic $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{29} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA24X2_IC : AE_SA24X2_IC_AE_FORMAT48<[]>;

class AE_SA24X2_IP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa24x2.ip $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA24X2_IP : AE_SA24X2_IP_AE_FORMAT48<[]>;

class AE_SA24X2_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa24x2.ric $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA24X2_RIC : AE_SA24X2_RIC_AE_FORMAT48<[]>;

class AE_SA24X2_RIP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa24x2.rip $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA24X2_RIP : AE_SA24X2_RIP_AE_FORMAT48<[]>;

class AE_SA32X2_IC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa32x2.ic $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{28} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA32X2_IC : AE_SA32X2_IC_AE_FORMAT48<[]>;

class AE_SA32X2_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa32x2.ip $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{4} = ae_ls_su{0};
let Inst{5} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA32X2_IP : AE_SA32X2_IP_X24<[]>;

class AE_SA32X2_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa32x2.ric $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA32X2_RIC : AE_SA32X2_RIC_AE_FORMAT48<[]>;

class AE_SA32X2_RIP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa32x2.rip $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{30} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA32X2_RIP : AE_SA32X2_RIP_AE_FORMAT48<[]>;

class AE_SA32X2F24_IC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa32x2f24.ic $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA32X2F24_IC : AE_SA32X2F24_IC_AE_FORMAT48<[]>;

class AE_SA32X2F24_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa32x2f24.ip $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{4} = ae_ls_su{0};
let Inst{5} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA32X2F24_IP : AE_SA32X2F24_IP_X24<[]>;

class AE_SA32X2F24_RIC_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa32x2f24.ric $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{28} = 1;
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA32X2F24_RIC : AE_SA32X2F24_RIC_AE_FORMAT48<[]>;

class AE_SA32X2F24_RIP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out, AR:$ars_out), (ins AE_DR:$ae_ls_v, AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa32x2f24.rip $ae_ls_v, $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_ls_v;
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out, $ars = $ars_out,@earlyclobber $ars_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{4} = 1;
let Inst{32} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{12} = ae_ls_v{0};
let Inst{13} = ae_ls_v{1};
let Inst{14} = ae_ls_v{2};
let Inst{15} = ae_ls_v{3};
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA32X2F24_RIP : AE_SA32X2F24_RIP_AE_FORMAT48<[]>;

class AE_SA64NEG_FP_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs AE_VALIGN:$ae_ls_su_out), (ins AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa64neg.fp $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out";

let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{5} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{15} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA64NEG_FP : AE_SA64NEG_FP_AE_FORMAT48<[(set AE_VALIGN:$ae_ls_su_out, (int_xtensa_ae_sa64neg_fp AE_VALIGN:$ae_ls_su, AR:$ars))]>;

class AE_SA64POS_FP_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_VALIGN:$ae_ls_su_out), (ins AE_VALIGN:$ae_ls_su, AR:$ars), "ae_sa64pos.fp $ae_ls_su, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_su;
bits<4> ars;
let Constraints = "$ae_ls_su = $ae_ls_su_out,@earlyclobber $ae_ls_su_out";

let mayStore = 1;
//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = ae_ls_su{0};
let Inst{5} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SA64POS_FP : AE_SA64POS_FP_X24<[(set AE_VALIGN:$ae_ls_su_out, (int_xtensa_ae_sa64pos_fp AE_VALIGN:$ae_ls_su, AR:$ars))]>;

class AE_SALIGN64_I_AE_FORMAT48<list<dag> pattern>
    : XtensaInst48<(outs ), (ins AE_VALIGN:$ae_ls_su, AR:$ars, imm64n_56:$ae_immls64), "ae_salign64.i $ae_ls_su, $ars, $ae_immls64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_ls_su;
bits<4> ars;
bits<4> ae_immls64;


let mayStore = 1;
//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//opcode
let Inst{14} = 1;
let Inst{30} = 1;
let Inst{33} = 1;
let Inst{34} = 1;
let Inst{37} = 1;
//operands
let Inst{6} = ae_ls_su{0};
let Inst{7} = ae_ls_su{1};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = ae_immls64{0};
let Inst{5} = ae_immls64{1};
let Inst{28} = ae_immls64{2};
let Inst{29} = ae_immls64{3};
}



def AE_SALIGN64_I : AE_SALIGN64_I_AE_FORMAT48<[(int_xtensa_ae_salign64_i AE_VALIGN:$ae_ls_su, AR:$ars, timm:$ae_immls64)]>;

class AE_SAT16X4_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_sat16x4 $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{56} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{28} = ae_arth_v0{0};
let Inst{29} = ae_arth_v0{1};
let Inst{30} = ae_arth_v0{2};
let Inst{31} = ae_arth_v0{3};
let Inst{36} = ae_arth_v1{0};
let Inst{37} = ae_arth_v1{1};
let Inst{38} = ae_arth_v1{2};
let Inst{39} = ae_arth_v1{3};
}



def AE_SAT16X4 : AE_SAT16X4_AE_FORMAT1<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_sat16x4 AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_SAT24S_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_sat24s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{29} = 1;
let Inst{30} = 1;
let Inst{31} = 1;
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{36} = ae_arth_v1{0};
let Inst{37} = ae_arth_v1{1};
let Inst{38} = ae_arth_v1{2};
let Inst{39} = ae_arth_v1{3};
}



def AE_SAT24S : AE_SAT24S_AE_FORMAT1<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_sat24s AE_DR:$ae_arth_v1))]>;

class AE_SAT48S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_sat48s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{11} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_SAT48S : AE_SAT48S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_sat48s AE_DR:$ae_arth_v1))]>;

class AE_SATQ56S_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v1), "ae_satq56s $ae_arth_v, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v1;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_arth_v{0};
let Inst{21} = ae_arth_v{1};
let Inst{22} = ae_arth_v{2};
let Inst{23} = ae_arth_v{3};
let Inst{36} = ae_arth_v1{0};
let Inst{37} = ae_arth_v1{1};
let Inst{38} = ae_arth_v1{2};
let Inst{39} = ae_arth_v1{3};
}



def AE_SATQ56S : AE_SATQ56S_AE_FORMAT1<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_satq56s AE_DR:$ae_arth_v1))]>;

class AE_SB_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, AR:$art), "ae_sb $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{13} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_SB : AE_SB_X24<[(set AR:$ars_out, (int_xtensa_ae_sb AR:$ars, AR:$art))]>;

class AE_SB_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, AR:$art), "ae_sb.ic $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_SB_IC : AE_SB_IC_X24<[(set AR:$ars_out, (int_xtensa_ae_sb_ic AR:$ars, AR:$art))]>;

class AE_SB_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, AR:$art), "ae_sb.ip $ars, $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> art;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_SB_IP : AE_SB_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_sb_ip AR:$ars, AR:$art))]>;

class AE_SBF_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars), "ae_sbf $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SBF : AE_SBF_X24<[(set AR:$ars_out, (int_xtensa_ae_sbf AR:$ars))]>;

class AE_SBF_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars), "ae_sbf.ic $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SBF_IC : AE_SBF_IC_X24<[(set AR:$ars_out, (int_xtensa_ae_sbf_ic AR:$ars))]>;

class AE_SBF_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars), "ae_sbf.ip $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{6} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SBF_IP : AE_SBF_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_sbf_ip AR:$ars))]>;

class AE_SBI_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, AR:$art, imm1_16:$ae_ohba2), "ae_sbi $ars, $art, $ae_ohba2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> art;
bits<4> ae_ohba2;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
let Inst{12} = ae_ohba2{0};
let Inst{13} = ae_ohba2{1};
let Inst{14} = ae_ohba2{2};
let Inst{15} = ae_ohba2{3};
}



def AE_SBI : AE_SBI_X24<[(set AR:$ars_out, (int_xtensa_ae_sbi AR:$ars, AR:$art, timm:$ae_ohba2))]>;

class AE_SBI_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, AR:$art, imm1_16:$ae_ohba2), "ae_sbi.ic $ars, $art, $ae_ohba2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> art;
bits<4> ae_ohba2;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
let Inst{12} = ae_ohba2{0};
let Inst{13} = ae_ohba2{1};
let Inst{14} = ae_ohba2{2};
let Inst{15} = ae_ohba2{3};
}



def AE_SBI_IC : AE_SBI_IC_X24<[(set AR:$ars_out, (int_xtensa_ae_sbi_ic AR:$ars, AR:$art, timm:$ae_ohba2))]>;

class AE_SBI_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars, AR:$art, imm1_16:$ae_ohba2), "ae_sbi.ip $ars, $art, $ae_ohba2", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
bits<4> art;
bits<4> ae_ohba2;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
let Inst{12} = ae_ohba2{0};
let Inst{13} = ae_ohba2{1};
let Inst{14} = ae_ohba2{2};
let Inst{15} = ae_ohba2{3};
}



def AE_SBI_IP : AE_SBI_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_sbi_ip AR:$ars, AR:$art, timm:$ae_ohba2))]>;

class AE_SEL16I_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_dr_to_dr_v), (ins AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1, uimm4:$ae_selimm), "ae_sel16i $ae_dr_to_dr_v, $ae_dr_to_dr_v0, $ae_dr_to_dr_v1, $ae_selimm", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_dr_to_dr_v;
bits<4> ae_dr_to_dr_v0;
bits<4> ae_dr_to_dr_v1;
bits<4> ae_selimm;



//format

//opcode
let Inst{2} = 1;
let Inst{21} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_dr_to_dr_v{0};
let Inst{13} = ae_dr_to_dr_v{1};
let Inst{14} = ae_dr_to_dr_v{2};
let Inst{15} = ae_dr_to_dr_v{3};
let Inst{8} = ae_dr_to_dr_v0{0};
let Inst{9} = ae_dr_to_dr_v0{1};
let Inst{10} = ae_dr_to_dr_v0{2};
let Inst{11} = ae_dr_to_dr_v0{3};
let Inst{4} = ae_dr_to_dr_v1{0};
let Inst{5} = ae_dr_to_dr_v1{1};
let Inst{6} = ae_dr_to_dr_v1{2};
let Inst{7} = ae_dr_to_dr_v1{3};
let Inst{16} = ae_selimm{0};
let Inst{17} = ae_selimm{1};
let Inst{18} = ae_selimm{2};
let Inst{19} = ae_selimm{3};
}



def AE_SEL16I : AE_SEL16I_X24<[(set AE_DR:$ae_dr_to_dr_v, (int_xtensa_ae_sel16i AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1, timm:$ae_selimm))]>;

class AE_SEL16I_N_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_dr_to_dr_v), (ins AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1, uimm2:$ae_selimm_N), "ae_sel16i.n $ae_dr_to_dr_v, $ae_dr_to_dr_v0, $ae_dr_to_dr_v1, $ae_selimm_N", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_dr_to_dr_v;
bits<4> ae_dr_to_dr_v0;
bits<4> ae_dr_to_dr_v1;
bits<2> ae_selimm_N;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode

//operands
let Inst{16} = ae_dr_to_dr_v{0};
let Inst{17} = ae_dr_to_dr_v{1};
let Inst{18} = ae_dr_to_dr_v{2};
let Inst{19} = ae_dr_to_dr_v{3};
let Inst{24} = ae_dr_to_dr_v0{0};
let Inst{25} = ae_dr_to_dr_v0{1};
let Inst{26} = ae_dr_to_dr_v0{2};
let Inst{27} = ae_dr_to_dr_v0{3};
let Inst{32} = ae_dr_to_dr_v1{0};
let Inst{33} = ae_dr_to_dr_v1{1};
let Inst{34} = ae_dr_to_dr_v1{2};
let Inst{35} = ae_dr_to_dr_v1{3};
let Inst{48} = ae_selimm_N{0};
let Inst{49} = ae_selimm_N{1};
}



def AE_SEL16I_N : AE_SEL16I_N_AE_FORMAT<[(set AE_DR:$ae_dr_to_dr_v, (int_xtensa_ae_sel16i_n AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1, timm:$ae_selimm_N))]>;

class AE_SEXT32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_dr_to_dr_v), (ins AE_DR:$ae_dr_to_dr_v0, seimm7_22:$ae_opnd_tp7), "ae_sext32 $ae_dr_to_dr_v, $ae_dr_to_dr_v0, $ae_opnd_tp7", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_dr_to_dr_v;
bits<4> ae_dr_to_dr_v0;
bits<4> ae_opnd_tp7;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
//operands
let Inst{12} = ae_dr_to_dr_v{0};
let Inst{13} = ae_dr_to_dr_v{1};
let Inst{14} = ae_dr_to_dr_v{2};
let Inst{15} = ae_dr_to_dr_v{3};
let Inst{8} = ae_dr_to_dr_v0{0};
let Inst{9} = ae_dr_to_dr_v0{1};
let Inst{10} = ae_dr_to_dr_v0{2};
let Inst{11} = ae_dr_to_dr_v0{3};
let Inst{4} = ae_opnd_tp7{0};
let Inst{5} = ae_opnd_tp7{1};
let Inst{6} = ae_opnd_tp7{2};
let Inst{7} = ae_opnd_tp7{3};
}



def AE_SEXT32 : AE_SEXT32_X24<[(set AE_DR:$ae_dr_to_dr_v, (int_xtensa_ae_sext32 AE_DR:$ae_dr_to_dr_v0, timm:$ae_opnd_tp7))]>;

class AE_SEXT32X2D16_10_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_to_dr_v), (ins AE_DR:$ae_to_dr_v0), "ae_sext32x2d16.10 $ae_to_dr_v, $ae_to_dr_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_to_dr_v;
bits<4> ae_to_dr_v0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{7} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{46} = 1;
let Inst{47} = 1;
//operands
let Inst{12} = ae_to_dr_v{0};
let Inst{13} = ae_to_dr_v{1};
let Inst{14} = ae_to_dr_v{2};
let Inst{15} = ae_to_dr_v{3};
let Inst{8} = ae_to_dr_v0{0};
let Inst{9} = ae_to_dr_v0{1};
let Inst{10} = ae_to_dr_v0{2};
let Inst{11} = ae_to_dr_v0{3};
}



def AE_SEXT32X2D16_10 : AE_SEXT32X2D16_10_AE_FORMAT<[(set AE_DR:$ae_to_dr_v, (int_xtensa_ae_sext32x2d16_10 AE_DR:$ae_to_dr_v0))]>;

class AE_SEXT32X2D16_32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_to_dr_v), (ins AE_DR:$ae_to_dr_v0), "ae_sext32x2d16.32 $ae_to_dr_v, $ae_to_dr_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_to_dr_v;
bits<4> ae_to_dr_v0;



//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
//operands
let Inst{12} = ae_to_dr_v{0};
let Inst{13} = ae_to_dr_v{1};
let Inst{14} = ae_to_dr_v{2};
let Inst{15} = ae_to_dr_v{3};
let Inst{8} = ae_to_dr_v0{0};
let Inst{9} = ae_to_dr_v0{1};
let Inst{10} = ae_to_dr_v0{2};
let Inst{11} = ae_to_dr_v0{3};
}



def AE_SEXT32X2D16_32 : AE_SEXT32X2D16_32_X24<[(set AE_DR:$ae_to_dr_v, (int_xtensa_ae_sext32x2d16_32 AE_DR:$ae_to_dr_v0))]>;

class AE_SHA32_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins AR:$ars), "ae_sha32 $arr, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SHA32 : AE_SHA32_X24<[(set AR:$arr, (int_xtensa_ae_sha32 AR:$ars))]>;

class AE_SHORTSWAP_AE_FORMAT1<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_to_dr_v), (ins AE_DR:$ae_to_dr_v0), "ae_shortswap $ae_to_dr_v, $ae_to_dr_v0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_to_dr_v;
bits<4> ae_to_dr_v0;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
let Inst{63} = 1;
//opcode
let Inst{36} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{56} = 1;
let Inst{59} = 1;
let Inst{62} = 1;
//operands
let Inst{20} = ae_to_dr_v{0};
let Inst{21} = ae_to_dr_v{1};
let Inst{22} = ae_to_dr_v{2};
let Inst{23} = ae_to_dr_v{3};
let Inst{28} = ae_to_dr_v0{0};
let Inst{29} = ae_to_dr_v0{1};
let Inst{30} = ae_to_dr_v0{2};
let Inst{31} = ae_to_dr_v0{3};
}



def AE_SHORTSWAP : AE_SHORTSWAP_AE_FORMAT1<[(set AE_DR:$ae_to_dr_v, (int_xtensa_ae_shortswap AE_DR:$ae_to_dr_v0))]>;

class AE_SLAA16S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_slaa16s $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SLAA16S : AE_SLAA16S_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slaa16s AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SLAA32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_slaa32 $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SLAA32 : AE_SLAA32_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slaa32 AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SLAA32S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_slaa32s $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SLAA32S : AE_SLAA32S_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slaa32s AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SLAA64_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_slaa64 $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SLAA64 : AE_SLAA64_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slaa64 AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SLAA64S_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_slaa64s $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{12} = 1;
let Inst{15} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{55} = 1;
//operands
let Inst{28} = ae_shift_d{0};
let Inst{29} = ae_shift_d{1};
let Inst{30} = ae_shift_d{2};
let Inst{31} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SLAA64S : AE_SLAA64S_AE_FORMAT2<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slaa64s AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SLAAQ56_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_slaaq56 $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SLAAQ56 : AE_SLAAQ56_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slaaq56 AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SLAI16S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm4:$ae_osa16), "ae_slai16s $ae_shift_d, $ae_shift_d0, $ae_osa16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ae_osa16;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
let Inst{36} = ae_osa16{0};
let Inst{37} = ae_osa16{1};
let Inst{38} = ae_osa16{2};
let Inst{39} = ae_osa16{3};
}



def AE_SLAI16S : AE_SLAI16S_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slai16s AE_DR:$ae_shift_d0, timm:$ae_osa16))]>;

class AE_SLAI24_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm5:$ae_osa32), "ae_slai24 $ae_shift_d, $ae_shift_d0, $ae_osa32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<5> ae_osa32;



//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ae_osa32{0};
let Inst{9} = ae_osa32{1};
let Inst{10} = ae_osa32{2};
let Inst{11} = ae_osa32{3};
let Inst{16} = ae_osa32{4};
}



def AE_SLAI24 : AE_SLAI24_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slai24 AE_DR:$ae_shift_d0, timm:$ae_osa32))]>;

class AE_SLAI24S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm5:$ae_osa32), "ae_slai24s $ae_shift_d, $ae_shift_d0, $ae_osa32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<5> ae_osa32;



//format

//opcode
let Inst{2} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ae_osa32{0};
let Inst{9} = ae_osa32{1};
let Inst{10} = ae_osa32{2};
let Inst{11} = ae_osa32{3};
let Inst{16} = ae_osa32{4};
}



def AE_SLAI24S : AE_SLAI24S_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slai24s AE_DR:$ae_shift_d0, timm:$ae_osa32))]>;

class AE_SLAI32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm5:$ae_osa32), "ae_slai32 $ae_shift_d, $ae_shift_d0, $ae_osa32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<5> ae_osa32;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ae_osa32{0};
let Inst{9} = ae_osa32{1};
let Inst{10} = ae_osa32{2};
let Inst{11} = ae_osa32{3};
let Inst{16} = ae_osa32{4};
}



def AE_SLAI32 : AE_SLAI32_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slai32 AE_DR:$ae_shift_d0, timm:$ae_osa32))]>;

class AE_SLAI32S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm5:$ae_osa32), "ae_slai32s $ae_shift_d, $ae_shift_d0, $ae_osa32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<5> ae_osa32;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ae_osa32{0};
let Inst{9} = ae_osa32{1};
let Inst{10} = ae_osa32{2};
let Inst{11} = ae_osa32{3};
let Inst{16} = ae_osa32{4};
}



def AE_SLAI32S : AE_SLAI32S_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slai32s AE_DR:$ae_shift_d0, timm:$ae_osa32))]>;

class AE_SLAI64_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm6:$ae_osa64), "ae_slai64 $ae_shift_d, $ae_shift_d0, $ae_osa64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<6> ae_osa64;



//format

//opcode
let Inst{2} = 1;
let Inst{19} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ae_osa64{0};
let Inst{9} = ae_osa64{1};
let Inst{10} = ae_osa64{2};
let Inst{11} = ae_osa64{3};
let Inst{16} = ae_osa64{4};
let Inst{17} = ae_osa64{5};
}



def AE_SLAI64 : AE_SLAI64_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slai64 AE_DR:$ae_shift_d0, timm:$ae_osa64))]>;

class AE_SLAI64S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm6:$ae_osa64), "ae_slai64s $ae_shift_d, $ae_shift_d0, $ae_osa64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<6> ae_osa64;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{58} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
let Inst{36} = ae_osa64{0};
let Inst{37} = ae_osa64{1};
let Inst{38} = ae_osa64{2};
let Inst{39} = ae_osa64{3};
let Inst{56} = ae_osa64{4};
let Inst{57} = ae_osa64{5};
}



def AE_SLAI64S : AE_SLAI64S_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slai64s AE_DR:$ae_shift_d0, timm:$ae_osa64))]>;

class AE_SLAISQ56S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm6:$ae_osa64), "ae_slaisq56s $ae_shift_d, $ae_shift_d0, $ae_osa64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<6> ae_osa64;



//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ae_osa64{0};
let Inst{9} = ae_osa64{1};
let Inst{10} = ae_osa64{2};
let Inst{11} = ae_osa64{3};
let Inst{16} = ae_osa64{4};
let Inst{17} = ae_osa64{5};
}



def AE_SLAISQ56S : AE_SLAISQ56S_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slaisq56s AE_DR:$ae_shift_d0, timm:$ae_osa64))]>;

class AE_SLAS24_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_slas24 $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{11} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
}



def AE_SLAS24 : AE_SLAS24_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slas24 AE_DR:$ae_shift_d0))]>;

class AE_SLAS24S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_slas24s $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
}



def AE_SLAS24S : AE_SLAS24S_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slas24s AE_DR:$ae_shift_d0))]>;

class AE_SLAS32_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_slas32 $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{37} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
}



def AE_SLAS32 : AE_SLAS32_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slas32 AE_DR:$ae_shift_d0))]>;

class AE_SLAS32S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_slas32s $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{37} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
}



def AE_SLAS32S : AE_SLAS32S_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slas32s AE_DR:$ae_shift_d0))]>;

class AE_SLAS64_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_slas64 $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{38} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
}



def AE_SLAS64 : AE_SLAS64_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slas64 AE_DR:$ae_shift_d0))]>;

class AE_SLAS64S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_slas64s $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
}



def AE_SLAS64S : AE_SLAS64S_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slas64s AE_DR:$ae_shift_d0))]>;

class AE_SLASQ56_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_slasq56 $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format

//opcode
let Inst{2} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
}



def AE_SLASQ56 : AE_SLASQ56_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slasq56 AE_DR:$ae_shift_d0))]>;

class AE_SLASSQ56S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_slassq56s $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
}



def AE_SLASSQ56S : AE_SLASSQ56S_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_slassq56s AE_DR:$ae_shift_d0))]>;

class AE_SRA64_32_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_sra64_32 $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{42} = 1;
let Inst{43} = 1;
let Inst{55} = 1;
//operands
let Inst{28} = ae_shift_d{0};
let Inst{29} = ae_shift_d{1};
let Inst{30} = ae_shift_d{2};
let Inst{31} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SRA64_32 : AE_SRA64_32_AE_FORMAT2<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_sra64_32 AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SRAA16RS_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_sraa16rs $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{55} = 1;
//operands
let Inst{28} = ae_shift_d{0};
let Inst{29} = ae_shift_d{1};
let Inst{30} = ae_shift_d{2};
let Inst{31} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SRAA16RS : AE_SRAA16RS_AE_FORMAT2<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_sraa16rs AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SRAA16S_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_sraa16s $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{14} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{55} = 1;
//operands
let Inst{28} = ae_shift_d{0};
let Inst{29} = ae_shift_d{1};
let Inst{30} = ae_shift_d{2};
let Inst{31} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SRAA16S : AE_SRAA16S_AE_FORMAT2<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_sraa16s AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SRAA32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_sraa32 $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SRAA32 : AE_SRAA32_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_sraa32 AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SRAA32RS_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_sraa32rs $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{55} = 1;
//operands
let Inst{28} = ae_shift_d{0};
let Inst{29} = ae_shift_d{1};
let Inst{30} = ae_shift_d{2};
let Inst{31} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SRAA32RS : AE_SRAA32RS_AE_FORMAT2<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_sraa32rs AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SRAA32S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_sraa32s $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SRAA32S : AE_SRAA32S_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_sraa32s AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SRAA64_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_sraa64 $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SRAA64 : AE_SRAA64_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_sraa64 AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SRAI16_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm4:$ae_osa16), "ae_srai16 $ae_shift_d, $ae_shift_d0, $ae_osa16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ae_osa16;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{56} = 1;
let Inst{57} = 1;
let Inst{58} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
let Inst{36} = ae_osa16{0};
let Inst{37} = ae_osa16{1};
let Inst{38} = ae_osa16{2};
let Inst{39} = ae_osa16{3};
}



def AE_SRAI16 : AE_SRAI16_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srai16 AE_DR:$ae_shift_d0, timm:$ae_osa16))]>;

class AE_SRAI16R_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm4:$ae_osa16), "ae_srai16r $ae_shift_d, $ae_shift_d0, $ae_osa16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ae_osa16;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
let Inst{36} = ae_osa16{0};
let Inst{37} = ae_osa16{1};
let Inst{38} = ae_osa16{2};
let Inst{39} = ae_osa16{3};
}



def AE_SRAI16R : AE_SRAI16R_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srai16r AE_DR:$ae_shift_d0, timm:$ae_osa16))]>;

class AE_SRAI24_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm5:$ae_osa32), "ae_srai24 $ae_shift_d, $ae_shift_d0, $ae_osa32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<5> ae_osa32;



//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ae_osa32{0};
let Inst{9} = ae_osa32{1};
let Inst{10} = ae_osa32{2};
let Inst{11} = ae_osa32{3};
let Inst{16} = ae_osa32{4};
}



def AE_SRAI24 : AE_SRAI24_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srai24 AE_DR:$ae_shift_d0, timm:$ae_osa32))]>;

class AE_SRAI32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm5:$ae_osa32), "ae_srai32 $ae_shift_d, $ae_shift_d0, $ae_osa32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<5> ae_osa32;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ae_osa32{0};
let Inst{9} = ae_osa32{1};
let Inst{10} = ae_osa32{2};
let Inst{11} = ae_osa32{3};
let Inst{16} = ae_osa32{4};
}



def AE_SRAI32 : AE_SRAI32_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srai32 AE_DR:$ae_shift_d0, timm:$ae_osa32))]>;

class AE_SRAI32R_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm5:$ae_osa32), "ae_srai32r $ae_shift_d, $ae_shift_d0, $ae_osa32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<5> ae_osa32;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{61} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
let Inst{36} = ae_osa32{0};
let Inst{37} = ae_osa32{1};
let Inst{38} = ae_osa32{2};
let Inst{39} = ae_osa32{3};
let Inst{56} = ae_osa32{4};
}



def AE_SRAI32R : AE_SRAI32R_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srai32r AE_DR:$ae_shift_d0, timm:$ae_osa32))]>;

class AE_SRAI64_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm6:$ae_osa64), "ae_srai64 $ae_shift_d, $ae_shift_d0, $ae_osa64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<6> ae_osa64;



//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ae_osa64{0};
let Inst{9} = ae_osa64{1};
let Inst{10} = ae_osa64{2};
let Inst{11} = ae_osa64{3};
let Inst{16} = ae_osa64{4};
let Inst{17} = ae_osa64{5};
}



def AE_SRAI64 : AE_SRAI64_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srai64 AE_DR:$ae_shift_d0, timm:$ae_osa64))]>;

class AE_SRAS24_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_sras24 $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
}



def AE_SRAS24 : AE_SRAS24_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_sras24 AE_DR:$ae_shift_d0))]>;

class AE_SRAS32_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_sras32 $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{39} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
}



def AE_SRAS32 : AE_SRAS32_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_sras32 AE_DR:$ae_shift_d0))]>;

class AE_SRAS64_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_sras64 $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{37} = 1;
let Inst{39} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
}



def AE_SRAS64 : AE_SRAS64_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_sras64 AE_DR:$ae_shift_d0))]>;

class AE_SRLA32_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_srla32 $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{55} = 1;
//operands
let Inst{28} = ae_shift_d{0};
let Inst{29} = ae_shift_d{1};
let Inst{30} = ae_shift_d{2};
let Inst{31} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SRLA32 : AE_SRLA32_AE_FORMAT2<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srla32 AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SRLA64_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AR:$ars), "ae_srla64 $ae_shift_d, $ae_shift_d0, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{12} = 1;
let Inst{40} = 1;
let Inst{43} = 1;
let Inst{44} = 1;
let Inst{55} = 1;
//operands
let Inst{28} = ae_shift_d{0};
let Inst{29} = ae_shift_d{1};
let Inst{30} = ae_shift_d{2};
let Inst{31} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_SRLA64 : AE_SRLA64_AE_FORMAT2<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srla64 AE_DR:$ae_shift_d0, AR:$ars))]>;

class AE_SRLI24_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm5:$ae_osa32), "ae_srli24 $ae_shift_d, $ae_shift_d0, $ae_osa32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<5> ae_osa32;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{57} = 1;
let Inst{61} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
let Inst{36} = ae_osa32{0};
let Inst{37} = ae_osa32{1};
let Inst{38} = ae_osa32{2};
let Inst{39} = ae_osa32{3};
let Inst{56} = ae_osa32{4};
}



def AE_SRLI24 : AE_SRLI24_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srli24 AE_DR:$ae_shift_d0, timm:$ae_osa32))]>;

class AE_SRLI32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm5:$ae_osa32), "ae_srli32 $ae_shift_d, $ae_shift_d0, $ae_osa32", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<5> ae_osa32;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{8} = ae_osa32{0};
let Inst{9} = ae_osa32{1};
let Inst{10} = ae_osa32{2};
let Inst{11} = ae_osa32{3};
let Inst{16} = ae_osa32{4};
}



def AE_SRLI32 : AE_SRLI32_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srli32 AE_DR:$ae_shift_d0, timm:$ae_osa32))]>;

class AE_SRLI64_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, uimm6:$ae_osa64), "ae_srli64 $ae_shift_d, $ae_shift_d0, $ae_osa64", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<6> ae_osa64;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{60} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
let Inst{36} = ae_osa64{0};
let Inst{37} = ae_osa64{1};
let Inst{38} = ae_osa64{2};
let Inst{39} = ae_osa64{3};
let Inst{56} = ae_osa64{4};
let Inst{57} = ae_osa64{5};
}



def AE_SRLI64 : AE_SRLI64_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srli64 AE_DR:$ae_shift_d0, timm:$ae_osa64))]>;

class AE_SRLS24_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_srls24 $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
}



def AE_SRLS24 : AE_SRLS24_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srls24 AE_DR:$ae_shift_d0))]>;

class AE_SRLS32_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_srls32 $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{36} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
}



def AE_SRLS32 : AE_SRLS32_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srls32 AE_DR:$ae_shift_d0))]>;

class AE_SRLS64_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0), "ae_srls64 $ae_shift_d, $ae_shift_d0", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{37} = 1;
let Inst{38} = 1;
let Inst{39} = 1;
let Inst{57} = 1;
let Inst{59} = 1;
let Inst{61} = 1;
let Inst{62} = 1;
let Inst{63} = 1;
//operands
let Inst{20} = ae_shift_d{0};
let Inst{21} = ae_shift_d{1};
let Inst{22} = ae_shift_d{2};
let Inst{23} = ae_shift_d{3};
let Inst{28} = ae_shift_d0{0};
let Inst{29} = ae_shift_d0{1};
let Inst{30} = ae_shift_d0{2};
let Inst{31} = ae_shift_d0{3};
}



def AE_SRLS64 : AE_SRLS64_AE_FORMAT<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_srls64 AE_DR:$ae_shift_d0))]>;

class AE_SUB16_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_sub16 $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{51} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{24} = ae_arth_v0{0};
let Inst{25} = ae_arth_v0{1};
let Inst{26} = ae_arth_v0{2};
let Inst{27} = ae_arth_v0{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_SUB16 : AE_SUB16_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_sub16 AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_SUB16S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_sub16s $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_SUB16S : AE_SUB16S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_sub16s AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_SUB24S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_sub24s $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{24} = ae_arth_v0{0};
let Inst{25} = ae_arth_v0{1};
let Inst{26} = ae_arth_v0{2};
let Inst{27} = ae_arth_v0{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_SUB24S : AE_SUB24S_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_sub24s AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_SUB32_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_sub32 $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_SUB32 : AE_SUB32_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_sub32 AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_SUB32S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_sub32s $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_SUB32S : AE_SUB32S_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_sub32s AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_SUB64_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_sub64 $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{22} = 1;
//operands
let Inst{12} = ae_arth_v{0};
let Inst{13} = ae_arth_v{1};
let Inst{14} = ae_arth_v{2};
let Inst{15} = ae_arth_v{3};
let Inst{8} = ae_arth_v0{0};
let Inst{9} = ae_arth_v0{1};
let Inst{10} = ae_arth_v0{2};
let Inst{11} = ae_arth_v0{3};
let Inst{4} = ae_arth_v1{0};
let Inst{5} = ae_arth_v1{1};
let Inst{6} = ae_arth_v1{2};
let Inst{7} = ae_arth_v1{3};
}



def AE_SUB64 : AE_SUB64_X24<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_sub64 AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_SUB64S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_sub64s $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{50} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{24} = ae_arth_v0{0};
let Inst{25} = ae_arth_v0{1};
let Inst{26} = ae_arth_v0{2};
let Inst{27} = ae_arth_v0{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_SUB64S : AE_SUB64S_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_sub64s AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_SUBADD32_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_subadd32 $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{24} = ae_arth_v0{0};
let Inst{25} = ae_arth_v0{1};
let Inst{26} = ae_arth_v0{2};
let Inst{27} = ae_arth_v0{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_SUBADD32 : AE_SUBADD32_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_subadd32 AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_SUBADD32S_AE_FORMAT<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_arth_v), (ins AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1), "ae_subadd32s $ae_arth_v, $ae_arth_v0, $ae_arth_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_arth_v;
bits<4> ae_arth_v0;
bits<4> ae_arth_v1;



//format
let Inst{0} = 1;
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{48} = 1;
let Inst{49} = 1;
let Inst{50} = 1;
let Inst{52} = 1;
let Inst{53} = 1;
let Inst{54} = 1;
let Inst{55} = 1;
//operands
let Inst{16} = ae_arth_v{0};
let Inst{17} = ae_arth_v{1};
let Inst{18} = ae_arth_v{2};
let Inst{19} = ae_arth_v{3};
let Inst{24} = ae_arth_v0{0};
let Inst{25} = ae_arth_v0{1};
let Inst{26} = ae_arth_v0{2};
let Inst{27} = ae_arth_v0{3};
let Inst{32} = ae_arth_v1{0};
let Inst{33} = ae_arth_v1{1};
let Inst{34} = ae_arth_v1{2};
let Inst{35} = ae_arth_v1{3};
}



def AE_SUBADD32S : AE_SUBADD32S_AE_FORMAT<[(set AE_DR:$ae_arth_v, (int_xtensa_ae_subadd32s AE_DR:$ae_arth_v0, AE_DR:$ae_arth_v1))]>;

class AE_TRUNCA32F64S_L_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AE_DR:$ae_shift_sd, AR:$ars), "ae_trunca32f64s.l $ae_shift_d, $ae_shift_d0, $ae_shift_sd, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ae_shift_sd;
bits<4> ars;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{41} = 1;
let Inst{55} = 1;
//operands
let Inst{28} = ae_shift_d{0};
let Inst{29} = ae_shift_d{1};
let Inst{30} = ae_shift_d{2};
let Inst{31} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{12} = ae_shift_sd{0};
let Inst{13} = ae_shift_sd{1};
let Inst{14} = ae_shift_sd{2};
let Inst{15} = ae_shift_sd{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_TRUNCA32F64S_L : AE_TRUNCA32F64S_L_AE_FORMAT2<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_trunca32f64s_l AE_DR:$ae_shift_d0, AE_DR:$ae_shift_sd, AR:$ars))]>;

class AE_TRUNCA32X2F64S_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AE_DR:$ae_shift_sd, AR:$ars), "ae_trunca32x2f64s $ae_shift_d, $ae_shift_d0, $ae_shift_sd, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ae_shift_sd;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = ae_shift_d{0};
let Inst{13} = ae_shift_d{1};
let Inst{14} = ae_shift_d{2};
let Inst{15} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{16} = ae_shift_sd{0};
let Inst{17} = ae_shift_sd{1};
let Inst{18} = ae_shift_sd{2};
let Inst{19} = ae_shift_sd{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_TRUNCA32X2F64S : AE_TRUNCA32X2F64S_X24<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_trunca32x2f64s AE_DR:$ae_shift_d0, AE_DR:$ae_shift_sd, AR:$ars))]>;

class AE_TRUNCI32F64S_L_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AE_DR:$ae_shift_sd, uimm4:$ae_osa16), "ae_trunci32f64s.l $ae_shift_d, $ae_shift_d0, $ae_shift_sd, $ae_osa16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ae_shift_sd;
bits<4> ae_osa16;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{40} = 1;
let Inst{41} = 1;
let Inst{42} = 1;
let Inst{55} = 1;
//operands
let Inst{28} = ae_shift_d{0};
let Inst{29} = ae_shift_d{1};
let Inst{30} = ae_shift_d{2};
let Inst{31} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{12} = ae_shift_sd{0};
let Inst{13} = ae_shift_sd{1};
let Inst{14} = ae_shift_sd{2};
let Inst{15} = ae_shift_sd{3};
let Inst{8} = ae_osa16{0};
let Inst{9} = ae_osa16{1};
let Inst{10} = ae_osa16{2};
let Inst{11} = ae_osa16{3};
}



def AE_TRUNCI32F64S_L : AE_TRUNCI32F64S_L_AE_FORMAT2<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_trunci32f64s_l AE_DR:$ae_shift_d0, AE_DR:$ae_shift_sd, timm:$ae_osa16))]>;

class AE_TRUNCI32X2F64S_AE_FORMAT2<list<dag> pattern>
    : XtensaInst64<(outs AE_DR:$ae_shift_d), (ins AE_DR:$ae_shift_d0, AE_DR:$ae_shift_sd, uimm4:$ae_osa16), "ae_trunci32x2f64s $ae_shift_d, $ae_shift_d0, $ae_shift_sd, $ae_osa16", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_shift_d;
bits<4> ae_shift_d0;
bits<4> ae_shift_sd;
bits<4> ae_osa16;



//format
let Inst{1} = 1;
let Inst{2} = 1;
let Inst{3} = 1;
//opcode
let Inst{43} = 1;
let Inst{55} = 1;
//operands
let Inst{28} = ae_shift_d{0};
let Inst{29} = ae_shift_d{1};
let Inst{30} = ae_shift_d{2};
let Inst{31} = ae_shift_d{3};
let Inst{4} = ae_shift_d0{0};
let Inst{5} = ae_shift_d0{1};
let Inst{6} = ae_shift_d0{2};
let Inst{7} = ae_shift_d0{3};
let Inst{12} = ae_shift_sd{0};
let Inst{13} = ae_shift_sd{1};
let Inst{14} = ae_shift_sd{2};
let Inst{15} = ae_shift_sd{3};
let Inst{8} = ae_osa16{0};
let Inst{9} = ae_osa16{1};
let Inst{10} = ae_osa16{2};
let Inst{11} = ae_osa16{3};
}



def AE_TRUNCI32X2F64S : AE_TRUNCI32X2F64S_AE_FORMAT2<[(set AE_DR:$ae_shift_d, (int_xtensa_ae_trunci32x2f64s AE_DR:$ae_shift_d0, AE_DR:$ae_shift_sd, timm:$ae_osa16))]>;

class AE_VLDL16C_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars), "ae_vldl16c $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_VLDL16C : AE_VLDL16C_X24<[(set AR:$ars_out, (int_xtensa_ae_vldl16c AR:$ars))]>;

class AE_VLDL16C_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars), "ae_vldl16c.ic $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_VLDL16C_IC : AE_VLDL16C_IC_X24<[(set AR:$ars_out, (int_xtensa_ae_vldl16c_ic AR:$ars))]>;

class AE_VLDL16C_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars), "ae_vldl16c.ip $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{7} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_VLDL16C_IP : AE_VLDL16C_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_vldl16c_ip AR:$ars))]>;

class AE_VLDL16T_X24<list<dag> pattern>
    : XtensaInst24<(outs BR:$br, AR:$art), (ins AR:$ars), "ae_vldl16t $br, $art, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> br;
bits<4> art;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
//operands
let Inst{12} = br{0};
let Inst{13} = br{1};
let Inst{14} = br{2};
let Inst{15} = br{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_VLDL16T : AE_VLDL16T_X24<[]>;

class AE_VLDL32T_X24<list<dag> pattern>
    : XtensaInst24<(outs BR:$br, AR:$art), (ins AR:$ars), "ae_vldl32t $br, $art, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> br;
bits<4> art;
bits<4> ars;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
//operands
let Inst{12} = br{0};
let Inst{13} = br{1};
let Inst{14} = br{2};
let Inst{15} = br{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_VLDL32T : AE_VLDL32T_X24<[]>;

class AE_VLDSHT_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "ae_vldsht $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{12} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def AE_VLDSHT : AE_VLDSHT_X24<[(int_xtensa_ae_vldsht AR:$art)]>;

class AE_VLEL16T_X24<list<dag> pattern>
    : XtensaInst24<(outs BR:$br, AR:$art_out), (ins AR:$art, AR:$ars), "ae_vlel16t $br, $art, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> br;
bits<4> art;
bits<4> ars;
let Constraints = "$art = $art_out,@earlyclobber $art_out";


//format

//opcode
let Inst{2} = 1;
let Inst{16} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = br{0};
let Inst{13} = br{1};
let Inst{14} = br{2};
let Inst{15} = br{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_VLEL16T : AE_VLEL16T_X24<[]>;

class AE_VLEL32T_X24<list<dag> pattern>
    : XtensaInst24<(outs BR:$br, AR:$art_out), (ins AR:$art, AR:$ars), "ae_vlel32t $br, $art, $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> br;
bits<4> art;
bits<4> ars;
let Constraints = "$art = $art_out,@earlyclobber $art_out";


//format

//opcode
let Inst{2} = 1;
let Inst{20} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = br{0};
let Inst{13} = br{1};
let Inst{14} = br{2};
let Inst{15} = br{3};
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_VLEL32T : AE_VLEL32T_X24<[]>;

class AE_VLES16C_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars), "ae_vles16c $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_VLES16C : AE_VLES16C_X24<[(set AR:$ars_out, (int_xtensa_ae_vles16c AR:$ars))]>;

class AE_VLES16C_IC_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars), "ae_vles16c.ic $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{5} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_VLES16C_IC : AE_VLES16C_IC_X24<[(set AR:$ars_out, (int_xtensa_ae_vles16c_ic AR:$ars))]>;

class AE_VLES16C_IP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$ars_out), (ins AR:$ars), "ae_vles16c.ip $ars", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ars;
let Constraints = "$ars = $ars_out,@earlyclobber $ars_out";


//format

//opcode
let Inst{2} = 1;
let Inst{6} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{8} = ars{0};
let Inst{9} = ars{1};
let Inst{10} = ars{2};
let Inst{11} = ars{3};
}



def AE_VLES16C_IP : AE_VLES16C_IP_X24<[(set AR:$ars_out, (int_xtensa_ae_vles16c_ip AR:$ars))]>;

class AE_XOR_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_DR:$ae_dr_to_dr_v), (ins AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1), "ae_xor $ae_dr_to_dr_v, $ae_dr_to_dr_v0, $ae_dr_to_dr_v1", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> ae_dr_to_dr_v;
bits<4> ae_dr_to_dr_v0;
bits<4> ae_dr_to_dr_v1;



//format

//opcode
let Inst{2} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{19} = 1;
let Inst{20} = 1;
//operands
let Inst{12} = ae_dr_to_dr_v{0};
let Inst{13} = ae_dr_to_dr_v{1};
let Inst{14} = ae_dr_to_dr_v{2};
let Inst{15} = ae_dr_to_dr_v{3};
let Inst{8} = ae_dr_to_dr_v0{0};
let Inst{9} = ae_dr_to_dr_v0{1};
let Inst{10} = ae_dr_to_dr_v0{2};
let Inst{11} = ae_dr_to_dr_v0{3};
let Inst{4} = ae_dr_to_dr_v1{0};
let Inst{5} = ae_dr_to_dr_v1{1};
let Inst{6} = ae_dr_to_dr_v1{2};
let Inst{7} = ae_dr_to_dr_v1{3};
}



def AE_XOR : AE_XOR_X24<[(set AE_DR:$ae_dr_to_dr_v, (int_xtensa_ae_xor AE_DR:$ae_dr_to_dr_v0, AE_DR:$ae_dr_to_dr_v1))]>;

class AE_ZALIGN64_X24<list<dag> pattern>
    : XtensaInst24<(outs AE_VALIGN:$ae_uu_uu), (ins ), "ae_zalign64 $ae_uu_uu", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<2> ae_uu_uu;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{12} = 1;
let Inst{19} = 1;
let Inst{21} = 1;
//operands
let Inst{6} = ae_uu_uu{0};
let Inst{7} = ae_uu_uu{1};
}



def AE_ZALIGN64 : AE_ZALIGN64_X24<[(set AE_VALIGN:$ae_uu_uu, (int_xtensa_ae_zalign64 ))]>;

class RUR_AE_BITHEAD_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins ), "rur.ae_bithead $arr", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;



//format

//opcode
let Inst{4} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
}



def RUR_AE_BITHEAD : RUR_AE_BITHEAD_X24<[(set AR:$arr, (int_xtensa_rur_ae_bithead ))]>;

class RUR_AE_BITPTR_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$art), (ins ), "rur.ae_bitptr $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def RUR_AE_BITPTR : RUR_AE_BITPTR_X24<[(set AR:$art, (int_xtensa_rur_ae_bitptr ))]>;

class RUR_AE_BITSUSED_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$art), (ins ), "rur.ae_bitsused $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def RUR_AE_BITSUSED : RUR_AE_BITSUSED_X24<[(set AR:$art, (int_xtensa_rur_ae_bitsused ))]>;

class RUR_AE_CBEGIN0_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins ), "rur.ae_cbegin0 $arr", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;



//format

//opcode
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
}



def RUR_AE_CBEGIN0 : RUR_AE_CBEGIN0_X24<[(set AR:$arr, (int_xtensa_rur_ae_cbegin0 ))]>;

class RUR_AE_CEND0_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins ), "rur.ae_cend0 $arr", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;



//format

//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{6} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
}



def RUR_AE_CEND0 : RUR_AE_CEND0_X24<[(set AR:$arr, (int_xtensa_rur_ae_cend0 ))]>;

class RUR_AE_CW_SD_NO_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins ), "rur.ae_cw_sd_no $arr", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;



//format

//opcode
let Inst{4} = 1;
let Inst{5} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
}



def RUR_AE_CW_SD_NO : RUR_AE_CW_SD_NO_X24<[(set AR:$arr, (int_xtensa_rur_ae_cw_sd_no ))]>;

class RUR_AE_CWRAP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$art), (ins ), "rur.ae_cwrap $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{11} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def RUR_AE_CWRAP : RUR_AE_CWRAP_X24<[(set AR:$art, (int_xtensa_rur_ae_cwrap ))]>;

class RUR_AE_FIRST_TS_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$art), (ins ), "rur.ae_first_ts $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{10} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def RUR_AE_FIRST_TS : RUR_AE_FIRST_TS_X24<[(set AR:$art, (int_xtensa_rur_ae_first_ts ))]>;

class RUR_AE_NEXTOFFSET_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$art), (ins ), "rur.ae_nextoffset $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def RUR_AE_NEXTOFFSET : RUR_AE_NEXTOFFSET_X24<[(set AR:$art, (int_xtensa_rur_ae_nextoffset ))]>;

class RUR_AE_OVERFLOW_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$art), (ins ), "rur.ae_overflow $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{11} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def RUR_AE_OVERFLOW : RUR_AE_OVERFLOW_X24<[(set AR:$art, (int_xtensa_rur_ae_overflow ))]>;

class RUR_AE_OVF_SAR_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins ), "rur.ae_ovf_sar $arr", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;



//format

//opcode
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
}



def RUR_AE_OVF_SAR : RUR_AE_OVF_SAR_X24<[(set AR:$arr, (int_xtensa_rur_ae_ovf_sar ))]>;

class RUR_AE_SAR_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$art), (ins ), "rur.ae_sar $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def RUR_AE_SAR : RUR_AE_SAR_X24<[(set AR:$art, (int_xtensa_rur_ae_sar ))]>;

class RUR_AE_SEARCHDONE_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$art), (ins ), "rur.ae_searchdone $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{11} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def RUR_AE_SEARCHDONE : RUR_AE_SEARCHDONE_X24<[(set AR:$art, (int_xtensa_rur_ae_searchdone ))]>;

class RUR_AE_TABLESIZE_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$art), (ins ), "rur.ae_tablesize $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{9} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def RUR_AE_TABLESIZE : RUR_AE_TABLESIZE_X24<[(set AR:$art, (int_xtensa_rur_ae_tablesize ))]>;

class RUR_AE_TS_FTS_BU_BP_X24<list<dag> pattern>
    : XtensaInst24<(outs AR:$arr), (ins ), "rur.ae_ts_fts_bu_bp $arr", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> arr;



//format

//opcode
let Inst{5} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{12} = arr{0};
let Inst{13} = arr{1};
let Inst{14} = arr{2};
let Inst{15} = arr{3};
}



def RUR_AE_TS_FTS_BU_BP : RUR_AE_TS_FTS_BU_BP_X24<[(set AR:$arr, (int_xtensa_rur_ae_ts_fts_bu_bp ))]>;

class WUR_AE_BITHEAD_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_bithead $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{8} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_BITHEAD : WUR_AE_BITHEAD_X24<[(int_xtensa_wur_ae_bithead AR:$art)]>;

class WUR_AE_BITPTR_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_bitptr $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_BITPTR : WUR_AE_BITPTR_X24<[(int_xtensa_wur_ae_bitptr AR:$art)]>;

class WUR_AE_BITSUSED_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_bitsused $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_BITSUSED : WUR_AE_BITSUSED_X24<[(int_xtensa_wur_ae_bitsused AR:$art)]>;

class WUR_AE_CBEGIN0_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_cbegin0 $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_CBEGIN0 : WUR_AE_CBEGIN0_X24<[(int_xtensa_wur_ae_cbegin0 AR:$art)]>;

class WUR_AE_CEND0_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_cend0 $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_CEND0 : WUR_AE_CEND0_X24<[(int_xtensa_wur_ae_cend0 AR:$art)]>;

class WUR_AE_CW_SD_NO_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_cw_sd_no $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_CW_SD_NO : WUR_AE_CW_SD_NO_X24<[(int_xtensa_wur_ae_cw_sd_no AR:$art)]>;

class WUR_AE_CWRAP_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_cwrap $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{11} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_CWRAP : WUR_AE_CWRAP_X24<[(int_xtensa_wur_ae_cwrap AR:$art)]>;

class WUR_AE_FIRST_TS_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_first_ts $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{10} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_FIRST_TS : WUR_AE_FIRST_TS_X24<[(int_xtensa_wur_ae_first_ts AR:$art)]>;

class WUR_AE_NEXTOFFSET_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_nextoffset $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{10} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_NEXTOFFSET : WUR_AE_NEXTOFFSET_X24<[(int_xtensa_wur_ae_nextoffset AR:$art)]>;

class WUR_AE_OVERFLOW_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_overflow $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{11} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_OVERFLOW : WUR_AE_OVERFLOW_X24<[(int_xtensa_wur_ae_overflow AR:$art)]>;

class WUR_AE_OVF_SAR_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_ovf_sar $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_OVF_SAR : WUR_AE_OVF_SAR_X24<[(int_xtensa_wur_ae_ovf_sar AR:$art)]>;

class WUR_AE_SAR_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_sar $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{10} = 1;
let Inst{11} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_SAR : WUR_AE_SAR_X24<[(int_xtensa_wur_ae_sar AR:$art)]>;

class WUR_AE_SEARCHDONE_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_searchdone $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{11} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_SEARCHDONE : WUR_AE_SEARCHDONE_X24<[(int_xtensa_wur_ae_searchdone AR:$art)]>;

class WUR_AE_TABLESIZE_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_tablesize $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{2} = 1;
let Inst{8} = 1;
let Inst{9} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{18} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_TABLESIZE : WUR_AE_TABLESIZE_X24<[(int_xtensa_wur_ae_tablesize AR:$art)]>;

class WUR_AE_TS_FTS_BU_BP_X24<list<dag> pattern>
    : XtensaInst24<(outs ), (ins AR:$art), "wur.ae_ts_fts_bu_bp $art", pattern>,
    Requires<[HasHIFI3]>
{
//operand decl
bits<4> art;



//format

//opcode
let Inst{9} = 1;
let Inst{12} = 1;
let Inst{13} = 1;
let Inst{14} = 1;
let Inst{15} = 1;
let Inst{16} = 1;
let Inst{17} = 1;
let Inst{20} = 1;
let Inst{21} = 1;
let Inst{22} = 1;
let Inst{23} = 1;
//operands
let Inst{4} = art{0};
let Inst{5} = art{1};
let Inst{6} = art{2};
let Inst{7} = art{3};
}



def WUR_AE_TS_FTS_BU_BP : WUR_AE_TS_FTS_BU_BP_X24<[(int_xtensa_wur_ae_ts_fts_bu_bp AR:$art)]>;
