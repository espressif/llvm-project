//===- RISCVInstrInfoH4.td - RISCV Target Description -*- tablegen -*------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISCV ESP32H4 DSP instructions in TableGen format.
//
// These definitions are generated
// This file is generated
//
//===----------------------------------------------------------------------===//

include "RISCVESP32H4Operands.td"
include "RISCVInstrFormatsESP32H4.td"

// This file is generated

def ESP_MULS16IX2: Esp32H4Inst<(outs GPRPIE:$rd, GPRPIE:$rd2), (ins GPRPIE:$rs1, GPRPIE:$rs2, shamt:$shamt),
                   "esp.muls16ix2\t $rd, $rd2, $rs1, $rs2, $shamt", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<6> shamt;
  bits<5> rd;
  bits<5> rd2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rd2{4-0};
  let Inst{26-25} = shamt{5-4};
  let Inst{24} = rs2{4};
  let Inst{23} = shamt{3};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = rs1{4};
  let Inst{18} = shamt{2};
  let Inst{17-15} = rs1{2-0};
  let Inst{14-13} = shamt{1-0};
  let Inst{12} = 0;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MULS16IX2_P : PseudoESP32H4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, shamt:$shamt, GPRPIE:$rd, GPRPIE:$rd2),
				"!esp_muls16ix2_p $rd, $rd2, $rs1, $rs2, $shamt",
			[(int_riscv_esp_muls16ix2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$shamt, GPRPIE:$rd, GPRPIE:$rd2)]>;

def ESP_MULS16X2: Esp32H4Inst<(outs GPR:$rd, GPR:$rd2), (ins GPR:$rs1, GPR:$rs2),
                   "esp.muls16x2\t $rd, $rd2, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  bits<5> rd2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rd2{4-0};
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MULS16X2_P : PseudoESP32H4<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rd, GPR:$rd2),
				"!esp_muls16x2_p $rd, $rd2, $rs1, $rs2",
			[(int_riscv_esp_muls16x2 GPR:$rs1, GPR:$rs2, GPR:$rd, GPR:$rd2)]>;

def ESP_MULS32I: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, shamt:$shamt),
                   "esp.muls32i\t $rd, $rs1, $rs2, $shamt", [(set GPR:$rd, (int_riscv_esp_muls32i GPR:$rs1, GPR:$rs2, timm:$shamt))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<6> shamt;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-26} = shamt{5-0};
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MULS32: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.muls32\t $rd, $rs1, $rs2", [(set GPR:$rd, (int_riscv_esp_muls32 GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_SHL: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.shl\t $rd, $rs1, $rs2", [(set GPR:$rd, (int_riscv_esp_shl GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_SHR: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.shr\t $rd, $rs1, $rs2", [(set GPR:$rd, (int_riscv_esp_shr GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_ABS: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs2),
                   "esp.abs\t $rd, $rs2", [(set GPR:$rd, (int_riscv_esp_abs GPR:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MOVR_XACC: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1),
                   "esp.movr.xacc\t $rd, $rs1", [(set GPR:$rd, (int_riscv_esp_movr_xacc GPR:$rs1))]> 
{
  bits<5> rs1;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MIN: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.min\t $rd, $rs1, $rs2", [(set GPR:$rd, (int_riscv_esp_min GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MAX: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.max\t $rd, $rs1, $rs2", [(set GPR:$rd, (int_riscv_esp_max GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_CLZS: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1),
                   "esp.clzs\t $rd, $rs1", [(set GPR:$rd, (int_riscv_esp_clzs GPR:$rs1))]> 
{
  bits<5> rs1;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_FLW_IP: Esp32H4Inst<(outs FPR32:$frd), (ins GPR:$rs1, imm5:$imm5, uimm2:$uimm2),
                   "esp.flw.ip\t $frd, $rs1, $imm5, $uimm2", [(set FPR32:$frd, (int_riscv_esp_flw_ip GPR:$rs1, timm:$imm5, timm:$uimm2))]> 
{
  bits<5> rs1;
  bits<5> imm5;
  bits<2> uimm2;
  bits<5> frd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26-25} = uimm2{1-0};
  let Inst{24-20} = imm5{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = frd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_FLW_XP: Esp32H4Inst<(outs FPR32:$frd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.flw.xp\t $frd, $rs1, $rs2", [(set FPR32:$frd, (int_riscv_esp_flw_xp GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> frd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = frd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_FSW_IP: Esp32H4Inst<(outs), (ins GPR:$rs1, FPR32:$frs, imm5:$imm5, uimm2:$uimm2),
                   "esp.fsw.ip\t $frs, $rs1, $imm5, $uimm2", [(int_riscv_esp_fsw_ip GPR:$rs1, FPR32:$frs, timm:$imm5, timm:$uimm2)]> 
{
  bits<5> rs1;
  bits<5> frs;
  bits<5> imm5;
  bits<2> uimm2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26-25} = uimm2{1-0};
  let Inst{24-20} = frs{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = imm5{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_FSW_XP: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs3, FPR32:$frs),
                   "esp.fsw.xp\t $frs, $rs1, $rs3", [(int_riscv_esp_fsw_xp GPR:$rs1, GPR:$rs3, FPR32:$frs)]> 
{
  bits<5> rs1;
  bits<5> rs3;
  bits<5> frs;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = frs{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LW_IP: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, imm5:$imm5, uimm2:$uimm2),
                   "esp.lw.ip\t $rd, $rs1, $imm5, $uimm2", [(set GPR:$rd, (int_riscv_esp_lw_ip GPR:$rs1, timm:$imm5, timm:$uimm2))]> 
{
  bits<5> rs1;
  bits<5> imm5;
  bits<2> uimm2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26-25} = uimm2{1-0};
  let Inst{24-20} = imm5{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LW_XP: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.lw.xp\t $rd, $rs1, $rs2", [(set GPR:$rd, (int_riscv_esp_lw_xp GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_SW_IP: Esp32H4Inst<(outs), (ins GPR:$rs3, GPR:$rs1, imm5:$imm5, uimm2:$uimm2),
                   "esp.sw.ip\t $rs3, $rs1, $imm5, $uimm2", [(int_riscv_esp_sw_ip GPR:$rs3, GPR:$rs1, timm:$imm5, timm:$uimm2)]> 
{
  bits<5> rs3;
  bits<5> rs1;
  bits<5> imm5;
  bits<2> uimm2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26-25} = uimm2{1-0};
  let Inst{24-20} = imm5{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_SW_XP: Esp32H4Inst<(outs), (ins GPR:$rs3, GPR:$rs1, GPR:$rs2),
                   "esp.sw.xp\t $rs3, $rs1, $rs2", [(int_riscv_esp_sw_xp GPR:$rs3, GPR:$rs1, GPR:$rs2)]> 
{
  bits<5> rs3;
  bits<5> rs1;
  bits<5> rs2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LH_IP: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, imm5:$imm5, uimm2:$uimm2),
                   "esp.lh.ip\t $rd, $rs1, $imm5, $uimm2", [(set GPR:$rd, (int_riscv_esp_lh_ip GPR:$rs1, timm:$imm5, timm:$uimm2))]> 
{
  bits<5> rs1;
  bits<5> imm5;
  bits<2> uimm2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26-25} = uimm2{1-0};
  let Inst{24-20} = imm5{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LH_XP: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.lh.xp\t $rd, $rs1, $rs2", [(set GPR:$rd, (int_riscv_esp_lh_xp GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_SH_IP: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs3, imm5:$imm5, uimm2:$uimm2),
                   "esp.sh.ip\t $rs3, $rs1, $imm5, $uimm2", [(int_riscv_esp_sh_ip GPR:$rs1, GPR:$rs3, timm:$imm5, timm:$uimm2)]> 
{
  bits<5> rs1;
  bits<5> rs3;
  bits<5> imm5;
  bits<2> uimm2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26-25} = uimm2{1-0};
  let Inst{24-20} = imm5{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_SH_XP: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rs3),
                   "esp.sh.xp\t $rs3, $rs1, $rs2", [(int_riscv_esp_sh_xp GPR:$rs1, GPR:$rs2, GPR:$rs3)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs3;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LB_IP: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, imm5:$imm5, uimm2:$uimm2),
                   "esp.lb.ip\t $rd, $rs1, $imm5, $uimm2", [(set GPR:$rd, (int_riscv_esp_lb_ip GPR:$rs1, timm:$imm5, timm:$uimm2))]> 
{
  bits<5> rs1;
  bits<5> imm5;
  bits<2> uimm2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26-25} = uimm2{1-0};
  let Inst{24-20} = imm5{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LB_XP: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.lb.xp\t $rd, $rs1, $rs2", [(set GPR:$rd, (int_riscv_esp_lb_xp GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_SB_IP: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs3, imm5:$imm5, uimm2:$uimm2),
                   "esp.sb.ip\t $rs3, $rs1, $imm5, $uimm2", [(int_riscv_esp_sb_ip GPR:$rs1, GPR:$rs3, timm:$imm5, timm:$uimm2)]> 
{
  bits<5> rs1;
  bits<5> rs3;
  bits<5> imm5;
  bits<2> uimm2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26-25} = uimm2{1-0};
  let Inst{24-20} = imm5{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_SB_XP: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rs3),
                   "esp.sb.xp\t $rs3, $rs1, $rs2", [(int_riscv_esp_sb_xp GPR:$rs1, GPR:$rs2, GPR:$rs3)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs3;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LHU_IP: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, imm5:$imm5, uimm2:$uimm2),
                   "esp.lhu.ip\t $rd, $rs1, $imm5, $uimm2", [(set GPR:$rd, (int_riscv_esp_lhu_ip GPR:$rs1, timm:$imm5, timm:$uimm2))]> 
{
  bits<5> rs1;
  bits<5> imm5;
  bits<2> uimm2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26-25} = uimm2{1-0};
  let Inst{24-20} = imm5{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LHU_XP: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.lhu.xp\t $rd, $rs1, $rs2", [(set GPR:$rd, (int_riscv_esp_lhu_xp GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LBU_IP: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, imm5:$imm5, uimm2:$uimm2),
                   "esp.lbu.ip\t $rd, $rs1, $imm5, $uimm2", [(set GPR:$rd, (int_riscv_esp_lbu_ip GPR:$rs1, timm:$imm5, timm:$uimm2))]> 
{
  bits<5> rs1;
  bits<5> imm5;
  bits<2> uimm2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26-25} = uimm2{1-0};
  let Inst{24-20} = imm5{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LBU_XP: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.lbu.xp\t $rd, $rs1, $rs2", [(set GPR:$rd, (int_riscv_esp_lbu_xp GPR:$rs1, GPR:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MOVW_XACC: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2),
                   "esp.movw.xacc\t $rs1, $rs2", [(int_riscv_esp_movw_xacc GPR:$rs1, GPR:$rs2)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MACS32I: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2, shamt:$shamt),
                   "esp.macs32i\t $rs1, $rs2, $shamt", [(int_riscv_esp_macs32i GPR:$rs1, GPR:$rs2, timm:$shamt)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<6> shamt;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-26} = shamt{5-0};
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MACS32: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2),
                   "esp.macs32\t $rs1, $rs2", [(int_riscv_esp_macs32 GPR:$rs1, GPR:$rs2)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MACS32_LD: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, GPR:$rs4),
                   "esp.macs32.ld\t $rd, $rs1, $rs2, $rs4", [(set GPR:$rd, (int_riscv_esp_macs32_ld GPR:$rs1, GPR:$rs2, GPR:$rs4))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs4;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rs4{4-0};
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MACS32_ST: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rs3, GPR:$rs4),
                   "esp.macs32.st\t $rs1, $rs2, $rs3, $rs4", [(int_riscv_esp_macs32_st GPR:$rs1, GPR:$rs2, GPR:$rs3, GPR:$rs4)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs3;
  bits<5> rs4;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rs4{4-0};
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MACS16IX2: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2, shamt:$shamt),
                   "esp.macs16ix2\t $rs1, $rs2, $shamt", [(int_riscv_esp_macs16ix2 GPR:$rs1, GPR:$rs2, timm:$shamt)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<6> shamt;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-26} = shamt{5-0};
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MACS16X2: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2),
                   "esp.macs16x2\t $rs1, $rs2", [(int_riscv_esp_macs16x2 GPR:$rs1, GPR:$rs2)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MACS16X2_LD: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, GPR:$rs4),
                   "esp.macs16x2.ld\t $rd, $rs1, $rs2, $rs4", [(set GPR:$rd, (int_riscv_esp_macs16x2_ld GPR:$rs1, GPR:$rs2, GPR:$rs4))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs4;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rs4{4-0};
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MACS16X1_LD: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, GPR:$rs4),
                   "esp.macs16x1.ld\t $rd, $rs1, $rs2, $rs4", [(set GPR:$rd, (int_riscv_esp_macs16x1_ld GPR:$rs1, GPR:$rs2, GPR:$rs4))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs4;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rs4{4-0};
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MACS16X2_ST: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rs3, GPR:$rs4),
                   "esp.macs16x2.st\t $rs1, $rs2, $rs3, $rs4", [(int_riscv_esp_macs16x2_st GPR:$rs1, GPR:$rs2, GPR:$rs3, GPR:$rs4)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs3;
  bits<5> rs4;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rs4{4-0};
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MSUS32I: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2, shamt:$shamt),
                   "esp.msus32i\t $rs1, $rs2, $shamt", [(int_riscv_esp_msus32i GPR:$rs1, GPR:$rs2, timm:$shamt)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<6> shamt;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-26} = shamt{5-0};
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MSUS32: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2),
                   "esp.msus32\t $rs1, $rs2", [(int_riscv_esp_msus32 GPR:$rs1, GPR:$rs2)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MSUS32_LD: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, GPR:$rs4),
                   "esp.msus32.ld\t $rd, $rs1, $rs2, $rs4", [(set GPR:$rd, (int_riscv_esp_msus32_ld GPR:$rs1, GPR:$rs2, GPR:$rs4))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs4;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rs4{4-0};
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MSUS32_ST: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rs3, GPR:$rs4),
                   "esp.msus32.st\t $rs1, $rs2, $rs3, $rs4", [(int_riscv_esp_msus32_st GPR:$rs1, GPR:$rs2, GPR:$rs3, GPR:$rs4)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs3;
  bits<5> rs4;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rs4{4-0};
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MSUS16IX2: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2, shamt:$shamt),
                   "esp.msus16ix2\t $rs1, $rs2, $shamt", [(int_riscv_esp_msus16ix2 GPR:$rs1, GPR:$rs2, timm:$shamt)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<6> shamt;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-26} = shamt{5-0};
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MSUS16X2: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2),
                   "esp.msus16x2\t $rs1, $rs2", [(int_riscv_esp_msus16x2 GPR:$rs1, GPR:$rs2)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MSUS16X2_LD: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, GPR:$rs4),
                   "esp.msus16x2.ld\t $rd, $rs1, $rs2, $rs4", [(set GPR:$rd, (int_riscv_esp_msus16x2_ld GPR:$rs1, GPR:$rs2, GPR:$rs4))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs4;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rs4{4-0};
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MSUS16X1_LD: Esp32H4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, GPR:$rs4),
                   "esp.msus16x1.ld\t $rd, $rs1, $rs2, $rs4", [(set GPR:$rd, (int_riscv_esp_msus16x1_ld GPR:$rs1, GPR:$rs2, GPR:$rs4))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs4;
  bits<5> rd;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rs4{4-0};
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_MSUS16X2_ST: Esp32H4Inst<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rs3, GPR:$rs4),
                   "esp.msus16x2.st\t $rs1, $rs2, $rs3, $rs4", [(int_riscv_esp_msus16x2_st GPR:$rs1, GPR:$rs2, GPR:$rs3, GPR:$rs4)]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs3;
  bits<5> rs4;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31-27} = rs4{4-0};
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-7} = rs3{4-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}
