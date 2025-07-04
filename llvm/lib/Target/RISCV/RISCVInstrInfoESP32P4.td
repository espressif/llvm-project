//===- RISCVInstrInfoP4.td - RISCV Target Description -*- tablegen -*------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISCV ESP32P4 DSP instructions in TableGen format.
//
// These definitions are generated
// This file is generated
//
//===----------------------------------------------------------------------===//

include "RISCVESP32P4Operands.td"
include "RISCVInstrFormatsESP32P4.td"
include "RISCVInstrInfoP4HWLP.td"

// This file is generated

def ESP_VCMULAS_S16_QACC_H: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vcmulas.s16.qacc.h\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = 1;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMULAS_S16_QACC_H_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vcmulas_s16_qacc_h_p $qx, $qy",
			[(int_riscv_esp_vcmulas_s16_qacc_h timm:$qx, timm:$qy)]>;

def ESP_VCMULAS_S16_QACC_H_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vcmulas.s16.qacc.h.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22-19} = off1616{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S16_QACC_H_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s16_qacc_h_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_h_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S16_QACC_H_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vcmulas.s16.qacc.h.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S16_QACC_H_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vcmulas_s16_qacc_h_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_h_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VCMULAS_S16_QACC_L: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vcmulas.s16.qacc.l\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = 1;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMULAS_S16_QACC_L_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vcmulas_s16_qacc_l_p $qx, $qy",
			[(int_riscv_esp_vcmulas_s16_qacc_l timm:$qx, timm:$qy)]>;

def ESP_VCMULAS_S16_QACC_L_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vcmulas.s16.qacc.l.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22-19} = off1616{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S16_QACC_L_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s16_qacc_l_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_l_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S16_QACC_L_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vcmulas.s16.qacc.l.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S16_QACC_L_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vcmulas_s16_qacc_l_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_l_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VCMULAS_S8_QACC_H: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vcmulas.s8.qacc.h\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 1;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMULAS_S8_QACC_H_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vcmulas_s8_qacc_h_p $qx, $qy",
			[(int_riscv_esp_vcmulas_s8_qacc_h timm:$qx, timm:$qy)]>;

def ESP_VCMULAS_S8_QACC_H_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vcmulas.s8.qacc.h.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-19} = off1616{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S8_QACC_H_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s8_qacc_h_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_h_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S8_QACC_H_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vcmulas.s8.qacc.h.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S8_QACC_H_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vcmulas_s8_qacc_h_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_h_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VCMULAS_S8_QACC_L: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vcmulas.s8.qacc.l\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 1;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMULAS_S8_QACC_L_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vcmulas_s8_qacc_l_p $qx, $qy",
			[(int_riscv_esp_vcmulas_s8_qacc_l timm:$qx, timm:$qy)]>;

def ESP_VCMULAS_S8_QACC_L_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vcmulas.s8.qacc.l.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-19} = off1616{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S8_QACC_L_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s8_qacc_l_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_l_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S8_QACC_L_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vcmulas.s8.qacc.l.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S8_QACC_L_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vcmulas_s8_qacc_l_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_l_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_S16_QACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vmulas.s16.qacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_S16_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_s16_qacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_s16_qacc timm:$qx, timm:$qy)]>;

def ESP_VMULAS_S16_QACC_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s16.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_QACC_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s16_qacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_S16_QACC_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s16.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_QACC_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s16_qacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_S16_QACC_ST_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s16.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_QACC_ST_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_s16_qacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

def ESP_VMULAS_S16_QACC_ST_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.s16.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_QACC_ST_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_s16_qacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VMULAS_S16_XACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vmulas.s16.xacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_S16_XACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_s16_xacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_s16_xacc timm:$qx, timm:$qy)]>;

def ESP_VMULAS_S16_XACC_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s16.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_XACC_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s16_xacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_S16_XACC_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s16.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_XACC_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s16_xacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_S16_XACC_ST_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s16.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_XACC_ST_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_s16_xacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

def ESP_VMULAS_S16_XACC_ST_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.s16.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_XACC_ST_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_s16_xacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VMULAS_S8_QACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vmulas.s8.qacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_S8_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_s8_qacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_s8_qacc timm:$qx, timm:$qy)]>;

def ESP_VMULAS_S8_QACC_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s8.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_QACC_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s8_qacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_S8_QACC_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s8.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_QACC_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s8_qacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_S8_QACC_ST_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s8.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_QACC_ST_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_s8_qacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

def ESP_VMULAS_S8_QACC_ST_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.s8.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_QACC_ST_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_s8_qacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VMULAS_S8_XACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vmulas.s8.xacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_S8_XACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_s8_xacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_s8_xacc timm:$qx, timm:$qy)]>;

def ESP_VMULAS_S8_XACC_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s8.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_XACC_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s8_xacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_S8_XACC_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s8.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_XACC_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s8_xacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_S8_XACC_ST_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s8.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_XACC_ST_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_s8_xacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

def ESP_VMULAS_S8_XACC_ST_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.s8.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_XACC_ST_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_s8_xacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VMULAS_U16_QACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vmulas.u16.qacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_U16_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_u16_qacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_u16_qacc timm:$qx, timm:$qy)]>;

def ESP_VMULAS_U16_QACC_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u16.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_QACC_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u16_qacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_U16_QACC_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u16.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_QACC_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u16_qacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_U16_QACC_ST_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u16.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_QACC_ST_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_u16_qacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

def ESP_VMULAS_U16_QACC_ST_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.u16.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_QACC_ST_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_u16_qacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VMULAS_U16_XACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vmulas.u16.xacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_U16_XACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_u16_xacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_u16_xacc timm:$qx, timm:$qy)]>;

def ESP_VMULAS_U16_XACC_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u16.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_XACC_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u16_xacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_U16_XACC_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u16.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_XACC_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u16_xacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_U16_XACC_ST_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u16.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_XACC_ST_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_u16_xacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

def ESP_VMULAS_U16_XACC_ST_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.u16.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_XACC_ST_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_u16_xacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VMULAS_U8_QACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vmulas.u8.qacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_U8_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_u8_qacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_u8_qacc timm:$qx, timm:$qy)]>;

def ESP_VMULAS_U8_QACC_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u8.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_QACC_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u8_qacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_U8_QACC_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u8.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_QACC_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u8_qacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_U8_QACC_ST_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u8.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_QACC_ST_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_u8_qacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

def ESP_VMULAS_U8_QACC_ST_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.u8.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_QACC_ST_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_u8_qacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VMULAS_U8_XACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy),
                   "esp.vmulas.u8.xacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_U8_XACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_u8_xacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_u8_xacc timm:$qx, timm:$qy)]>;

def ESP_VMULAS_U8_XACC_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u8.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_XACC_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u8_xacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_U8_XACC_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u8.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_XACC_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u8_xacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_U8_XACC_ST_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u8.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21-19} = off1616{3-1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off1616{0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_XACC_ST_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_u8_xacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

def ESP_VMULAS_U8_XACC_ST_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.u8.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_XACC_ST_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_u8_xacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VMULAS_S16_QACC_LDBC_INCP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s16.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_QACC_LDBC_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s16_qacc_ldbc_incp_p $qu, $rs1, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_ldbc_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_S8_QACC_LDBC_INCP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s8.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_QACC_LDBC_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s8_qacc_ldbc_incp_p $qu, $rs1, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_ldbc_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_U16_QACC_LDBC_INCP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u16.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_QACC_LDBC_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u16_qacc_ldbc_incp_p $qu, $rs1, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_ldbc_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VMULAS_U8_QACC_LDBC_INCP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u8.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_QACC_LDBC_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u8_qacc_ldbc_incp_p $qu, $rs1, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_ldbc_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VSMULAS_S16_QACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy, select_16:$sel16),
                   "esp.vsmulas.s16.qacc\t $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSMULAS_S16_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, select_16:$sel16),
				"!esp_vsmulas_s16_qacc_p $qx, $qy, $sel16",
			[(int_riscv_esp_vsmulas_s16_qacc timm:$qx, timm:$qy, timm:$sel16)]>;

def ESP_VSMULAS_S16_QACC_LD_INCP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, select_16:$sel16),
                   "esp.vsmulas.s16.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22-19} = sel16{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSMULAS_S16_QACC_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_s16_qacc_ld_incp_p $qu, $rs1, $qx, $qy, $sel16",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_s16_qacc_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel16, timm:$qu))]>;

def ESP_VSMULAS_S8_QACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy, select_16:$sel16),
                   "esp.vsmulas.s8.qacc\t $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSMULAS_S8_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, select_16:$sel16),
				"!esp_vsmulas_s8_qacc_p $qx, $qy, $sel16",
			[(int_riscv_esp_vsmulas_s8_qacc timm:$qx, timm:$qy, timm:$sel16)]>;

def ESP_VSMULAS_S8_QACC_LD_INCP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, select_16:$sel16),
                   "esp.vsmulas.s8.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22-19} = sel16{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSMULAS_S8_QACC_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_s8_qacc_ld_incp_p $qu, $rs1, $qx, $qy, $sel16",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_s8_qacc_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel16, timm:$qu))]>;

def ESP_VSMULAS_U16_QACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy, select_16:$sel16),
                   "esp.vsmulas.u16.qacc\t $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSMULAS_U16_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, select_16:$sel16),
				"!esp_vsmulas_u16_qacc_p $qx, $qy, $sel16",
			[(int_riscv_esp_vsmulas_u16_qacc timm:$qx, timm:$qy, timm:$sel16)]>;

def ESP_VSMULAS_U16_QACC_LD_INCP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, select_16:$sel16),
                   "esp.vsmulas.u16.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-19} = sel16{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSMULAS_U16_QACC_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_u16_qacc_ld_incp_p $qu, $rs1, $qx, $qy, $sel16",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_u16_qacc_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel16, timm:$qu))]>;

def ESP_VSMULAS_U8_QACC: Esp32P4Inst<(outs), (ins QR:$qx, QR:$qy, select_16:$sel16),
                   "esp.vsmulas.u8.qacc\t $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSMULAS_U8_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, select_16:$sel16),
				"!esp_vsmulas_u8_qacc_p $qx, $qy, $sel16",
			[(int_riscv_esp_vsmulas_u8_qacc timm:$qx, timm:$qy, timm:$sel16)]>;

def ESP_VSMULAS_U8_QACC_LD_INCP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, select_16:$sel16),
                   "esp.vsmulas.u8.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-19} = sel16{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSMULAS_U8_QACC_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_u8_qacc_ld_incp_p $qu, $rs1, $qx, $qy, $sel16",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_u8_qacc_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel16, timm:$qu))]>;

def ESP_CMUL_S16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, select_4:$sel4),
                   "esp.cmul.s16\t $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16-15} = sel4{1-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_CMUL_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, select_4:$sel4, imm8:$qz),
				"!esp_cmul_s16_p $qz, $qx, $qy, $sel4",
			[(int_riscv_esp_cmul_s16 timm:$qx, timm:$qy, timm:$sel4, timm:$qz)]>;

def ESP_CMUL_S16_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4),
                   "esp.cmul.s16.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_CMUL_S16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_4:$sel4, imm8:$qz, imm8:$qu),
				"!esp_cmul_s16_ld_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel4, timm:$qz, timm:$qu))]>;

def ESP_CMUL_S16_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4),
                   "esp.cmul.s16.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_CMUL_S16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, select_4:$sel4, imm8:$qz),
				"!esp_cmul_s16_st_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sel4, timm:$qz))]>;

def ESP_CMUL_S8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, select_4:$sel4),
                   "esp.cmul.s8\t $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16-15} = sel4{1-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_CMUL_S8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, select_4:$sel4, imm8:$qz),
				"!esp_cmul_s8_p $qz, $qx, $qy, $sel4",
			[(int_riscv_esp_cmul_s8 timm:$qx, timm:$qy, timm:$sel4, timm:$qz)]>;

def ESP_CMUL_S8_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4),
                   "esp.cmul.s8.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_CMUL_S8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_4:$sel4, imm8:$qz, imm8:$qu),
				"!esp_cmul_s8_ld_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel4, timm:$qz, timm:$qu))]>;

def ESP_CMUL_S8_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4),
                   "esp.cmul.s8.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_CMUL_S8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, select_4:$sel4, imm8:$qz),
				"!esp_cmul_s8_st_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sel4, timm:$qz))]>;

def ESP_CMUL_U16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, select_4:$sel4),
                   "esp.cmul.u16\t $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16-15} = sel4{1-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_CMUL_U16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, select_4:$sel4, imm8:$qz),
				"!esp_cmul_u16_p $qz, $qx, $qy, $sel4",
			[(int_riscv_esp_cmul_u16 timm:$qx, timm:$qy, timm:$sel4, timm:$qz)]>;

def ESP_CMUL_U16_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4),
                   "esp.cmul.u16.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_CMUL_U16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_4:$sel4, imm8:$qz, imm8:$qu),
				"!esp_cmul_u16_ld_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel4, timm:$qz, timm:$qu))]>;

def ESP_CMUL_U16_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4),
                   "esp.cmul.u16.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_CMUL_U16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, select_4:$sel4, imm8:$qz),
				"!esp_cmul_u16_st_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sel4, timm:$qz))]>;

def ESP_CMUL_U8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, select_4:$sel4),
                   "esp.cmul.u8\t $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16-15} = sel4{1-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_CMUL_U8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, select_4:$sel4, imm8:$qz),
				"!esp_cmul_u8_p $qz, $qx, $qy, $sel4",
			[(int_riscv_esp_cmul_u8 timm:$qx, timm:$qy, timm:$sel4, timm:$qz)]>;

def ESP_CMUL_U8_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4),
                   "esp.cmul.u8.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_CMUL_U8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_4:$sel4, imm8:$qz, imm8:$qu),
				"!esp_cmul_u8_ld_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel4, timm:$qz, timm:$qu))]>;

def ESP_CMUL_U8_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4),
                   "esp.cmul.u8.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_CMUL_U8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, select_4:$sel4, imm8:$qz),
				"!esp_cmul_u8_st_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sel4, timm:$qz))]>;

def ESP_MAX_S16_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.s16.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_S16_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_s16_a_p $qw, $rd",
			[(int_riscv_esp_max_s16_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_S32_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.s32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_S32_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_s32_a_p $qw, $rd",
			[(int_riscv_esp_max_s32_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_S8_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.s8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_S8_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_s8_a_p $qw, $rd",
			[(int_riscv_esp_max_s8_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_U16_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.u16.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_U16_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_u16_a_p $qw, $rd",
			[(int_riscv_esp_max_u16_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_U32_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.u32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_U32_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_u32_a_p $qw, $rd",
			[(int_riscv_esp_max_u32_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_U8_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.u8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_U8_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_u8_a_p $qw, $rd",
			[(int_riscv_esp_max_u8_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_S16_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.s16.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_S16_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_s16_a_p $qw, $rd",
			[(int_riscv_esp_min_s16_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_S32_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.s32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_S32_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_s32_a_p $qw, $rd",
			[(int_riscv_esp_min_s32_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_S8_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.s8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_S8_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_s8_a_p $qw, $rd",
			[(int_riscv_esp_min_s8_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_U16_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.u16.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_U16_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_u16_a_p $qw, $rd",
			[(int_riscv_esp_min_u16_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_U32_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.u32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_U32_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_u32_a_p $qw, $rd",
			[(int_riscv_esp_min_u32_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_U8_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.u8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_U8_A_P : PseudoESP32P4<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_u8_a_p $qw, $rd",
			[(int_riscv_esp_min_u8_a timm:$qw, GPRPIE:$rd)]>;

def ESP_VABS_16: Esp32P4Inst<(outs QR:$qv), (ins QR:$qy),
                   "esp.vabs.16\t $qv, $qy", []> 
{
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
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

let usesCustomInserter = 1 in 
def ESP_VABS_16_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qv),
				"!esp_vabs_16_p $qv, $qy",
			[(int_riscv_esp_vabs_16 timm:$qy, timm:$qv)]>;

def ESP_VABS_32: Esp32P4Inst<(outs QR:$qv), (ins QR:$qy),
                   "esp.vabs.32\t $qv, $qy", []> 
{
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 1;
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

let usesCustomInserter = 1 in 
def ESP_VABS_32_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qv),
				"!esp_vabs_32_p $qv, $qy",
			[(int_riscv_esp_vabs_32 timm:$qy, timm:$qv)]>;

def ESP_VABS_8: Esp32P4Inst<(outs QR:$qv), (ins QR:$qy),
                   "esp.vabs.8\t $qv, $qy", []> 
{
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
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

let usesCustomInserter = 1 in 
def ESP_VABS_8_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qv),
				"!esp_vabs_8_p $qv, $qy",
			[(int_riscv_esp_vabs_8 timm:$qy, timm:$qv)]>;

def ESP_VADD_S16: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.s16\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_s16_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_s16 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VADD_S16_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.s16.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_s16_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_S16_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.s16.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_s16_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VADD_S32: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.s32\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_S32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_s32_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_s32 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VADD_S32_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.s32.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S32_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_s32_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_S32_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.s32.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S32_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_s32_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VADD_S8: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.s8\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_S8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_s8_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_s8 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VADD_S8_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.s8.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_s8_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_S8_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.s8.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_s8_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VADD_U16: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.u16\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_U16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_u16_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_u16 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VADD_U16_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.u16.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_u16_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_U16_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.u16.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_u16_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VADD_U32: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.u32\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_U32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_u32_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_u32 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VADD_U32_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.u32.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U32_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_u32_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_U32_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.u32.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U32_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_u32_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VADD_U8: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.u8\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_U8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_u8_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_u8 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VADD_U8_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.u8.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_u8_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_U8_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.u8.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_u8_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VCLAMP_S16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, select_16:$sel16),
                   "esp.vclamp.s16\t $qz, $qx, $sel16", []> 
{
  bits<3> qx;
  bits<4> sel16;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = sel16{3-2};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19-18} = sel16{1-0};
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCLAMP_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, select_16:$sel16, imm8:$qz),
				"!esp_vclamp_s16_p $qz, $qx, $sel16",
			[(int_riscv_esp_vclamp_s16 timm:$qx, timm:$sel16, timm:$qz)]>;

def ESP_VMAX_S16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.s16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_S16_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.s16.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_S16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_s16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_S16_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.s16.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_S16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_s16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMAX_S32: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.s32\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_S32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_s32_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_s32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_S32_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.s32.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_S32_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_s32_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_S32_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.s32.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_S32_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_s32_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMAX_S8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.s8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_S8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_S8_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.s8.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_S8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_s8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_S8_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.s8.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_S8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_s8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMAX_U16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.u16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_U16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_U16_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.u16.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_U16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_u16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_U16_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.u16.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_U16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_u16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMAX_U32: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.u32\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_U32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_u32_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_u32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_U32_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.u32.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_U32_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_u32_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_U32_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.u32.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_U32_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_u32_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMAX_U8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.u8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_U8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_u8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_U8_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.u8.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_U8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_u8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_U8_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.u8.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMAX_U8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_u8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_S16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.s16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_S16_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.s16.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_S16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_s16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_S16_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.s16.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_S16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_s16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_S32: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.s32\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_S32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_s32_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_s32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_S32_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.s32.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_S32_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_s32_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_S32_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.s32.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_S32_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_s32_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_S8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.s8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_S8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_S8_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.s8.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_S8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_s8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_S8_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.s8.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_S8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_s8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_U16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.u16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_U16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_U16_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.u16.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_U16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_u16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_U16_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.u16.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_U16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_u16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_U32: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.u32\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_U32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_u32_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_u32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_U32_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.u32.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_U32_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_u32_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_U32_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.u32.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_U32_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_u32_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_U8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.u8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_U8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_u8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_U8_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.u8.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_U8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_u8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_U8_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.u8.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMIN_U8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_u8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMUL_S16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmul.s16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmul_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmul_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMUL_S16_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmul.s16.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMUL_S16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmul_s16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMUL_S16_S8XS8: Esp32P4Inst<(outs QR:$qz, QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vmul.s16.s8xs8\t $qz, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_S16_S8XS8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz, imm8:$qv),
				"!esp_vmul_s16_s8xs8_p $qz, $qv, $qx, $qy",
			[(int_riscv_esp_vmul_s16_s8xs8 timm:$qx, timm:$qy, timm:$qz, timm:$qv)]>;

def ESP_VMUL_S16_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmul.s16.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMUL_S16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmul_s16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMUL_S32_S16XS16: Esp32P4Inst<(outs QR:$qz, QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vmul.s32.s16xs16\t $qz, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_S32_S16XS16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz, imm8:$qv),
				"!esp_vmul_s32_s16xs16_p $qz, $qv, $qx, $qy",
			[(int_riscv_esp_vmul_s32_s16xs16 timm:$qx, timm:$qy, timm:$qz, timm:$qv)]>;

def ESP_VMUL_S8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmul.s8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_S8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmul_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmul_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMUL_S8_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmul.s8.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMUL_S8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmul_s8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMUL_S8_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmul.s8.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMUL_S8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmul_s8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMUL_U16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmul.u16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_U16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmul_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmul_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMUL_U16_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmul.u16.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMUL_U16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmul_u16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMUL_U16_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmul.u16.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMUL_U16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmul_u16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMUL_U8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmul.u8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_U8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmul_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmul_u8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMUL_U8_LD_INCP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmul.u8.ld.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMUL_U8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmul_u8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMUL_U8_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmul.u8.st.incp\t $qu, $rs1, $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMUL_U8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmul_u8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VPRELU_S16: Esp32P4Inst<(outs QR:$qz), (ins GPRPIE:$rs1, QR:$qx, QR:$qy),
                   "esp.vprelu.s16\t $qz, $qy, $qx, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VPRELU_S16_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vprelu_s16_p $qz, $qy, $qx, $rs1",
			[(int_riscv_esp_vprelu_s16 GPRPIE:$rs1, timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VPRELU_S8: Esp32P4Inst<(outs QR:$qz), (ins GPRPIE:$rs1, QR:$qx, QR:$qy),
                   "esp.vprelu.s8\t $qz, $qy, $qx, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VPRELU_S8_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vprelu_s8_p $qz, $qy, $qx, $rs1",
			[(int_riscv_esp_vprelu_s8 GPRPIE:$rs1, timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VRELU_S16: Esp32P4Inst<(outs QR:$qyr), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy),
                   "esp.vrelu.s16\t $qy, $rs2, $rs1", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qy;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qyr = $qy";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 1;
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

let usesCustomInserter = 1 in 
def ESP_VRELU_S16_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy),
				"!esp_vrelu_s16_p $qy, $rs2, $rs1",
			[(int_riscv_esp_vrelu_s16 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy)]>;

def ESP_VRELU_S8: Esp32P4Inst<(outs QR:$qyr), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy),
                   "esp.vrelu.s8\t $qy, $rs2, $rs1", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qy;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qyr = $qy";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
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

let usesCustomInserter = 1 in 
def ESP_VRELU_S8_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy),
				"!esp_vrelu_s8_p $qy, $rs2, $rs1",
			[(int_riscv_esp_vrelu_s8 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy)]>;

def ESP_VSADDS_S16: Esp32P4Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vsadds.s16\t $qv, $qx, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSADDS_S16_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vsadds_s16_p $qv, $qx, $rs1",
			[(int_riscv_esp_vsadds_s16 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSADDS_S8: Esp32P4Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vsadds.s8\t $qv, $qx, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSADDS_S8_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vsadds_s8_p $qv, $qx, $rs1",
			[(int_riscv_esp_vsadds_s8 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSADDS_U16: Esp32P4Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vsadds.u16\t $qv, $qx, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSADDS_U16_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vsadds_u16_p $qv, $qx, $rs1",
			[(int_riscv_esp_vsadds_u16 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSADDS_U8: Esp32P4Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vsadds.u8\t $qv, $qx, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSADDS_U8_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vsadds_u8_p $qv, $qx, $rs1",
			[(int_riscv_esp_vsadds_u8 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSAT_S16: Esp32P4Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.s16\t $qz, $qx, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_S16_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_s16_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_s16 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_S32: Esp32P4Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.s32\t $qz, $qx, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_S32_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_s32_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_s32 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_S8: Esp32P4Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.s8\t $qz, $qx, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_S8_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_s8_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_s8 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_U16: Esp32P4Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.u16\t $qz, $qx, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_U16_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_u16_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_u16 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_U32: Esp32P4Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.u32\t $qz, $qx, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_U32_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_u32_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_u32 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_U8: Esp32P4Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.u8\t $qz, $qx, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_U8_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_u8_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_u8 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSSUBS_S16: Esp32P4Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vssubs.s16\t $qv, $qx, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSSUBS_S16_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vssubs_s16_p $qv, $qx, $rs1",
			[(int_riscv_esp_vssubs_s16 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSSUBS_S8: Esp32P4Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vssubs.s8\t $qv, $qx, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSSUBS_S8_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vssubs_s8_p $qv, $qx, $rs1",
			[(int_riscv_esp_vssubs_s8 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSSUBS_U16: Esp32P4Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vssubs.u16\t $qv, $qx, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSSUBS_U16_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vssubs_u16_p $qv, $qx, $rs1",
			[(int_riscv_esp_vssubs_u16 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSSUBS_U8: Esp32P4Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vssubs.u8\t $qv, $qx, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSSUBS_U8_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vssubs_u8_p $qv, $qx, $rs1",
			[(int_riscv_esp_vssubs_u8 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSUB_S16: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.s16\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_s16_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_s16 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_S16_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.s16.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_s16_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_S16_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.s16.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_s16_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VSUB_S32: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.s32\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_S32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_s32_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_s32 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_S32_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.s32.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S32_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_s32_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_S32_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.s32.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S32_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_s32_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VSUB_S8: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.s8\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_S8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_s8_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_s8 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_S8_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.s8.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_s8_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_S8_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.s8.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_s8_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VSUB_U16: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.u16\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_U16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_u16_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_u16 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_U16_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.u16.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_u16_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_U16_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.u16.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_u16_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VSUB_U32: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.u32\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_U32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_u32_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_u32 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_U32_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.u32.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U32_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_u32_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_U32_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.u32.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U32_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_u32_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_VSUB_U8: Esp32P4Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.u8\t $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_U8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_u8_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_u8 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_U8_LD_INCP: Esp32P4Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.u8.ld.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U8_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_u8_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_U8_ST_INCP: Esp32P4Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.u8.st.incp\t $qu, $rs1, $qv, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U8_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_u8_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;

def ESP_ADDX2: Esp32P4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.addx2\t $rd, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 0;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_ADDX2_P : PseudoESP32P4<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rd),
				"!esp_addx2_p $rd, $rs1, $rs2",
			[(int_riscv_esp_addx2 GPR:$rs1, GPR:$rs2, GPR:$rd)]>;

def ESP_ADDX4: Esp32P4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.addx4\t $rd, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
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
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 0;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_ADDX4_P : PseudoESP32P4<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rd),
				"!esp_addx4_p $rd, $rs1, $rs2",
			[(int_riscv_esp_addx4 GPR:$rs1, GPR:$rs2, GPR:$rd)]>;

def ESP_SAT: Esp32P4Inst<(outs GPR:$rsdr), (ins GPR:$rs0, GPR:$rs1, GPR:$rsd),
                   "esp.sat\t $rsd, $rs0, $rs1", []> 
{
  bits<5> rs0;
  bits<5> rs1;
  bits<5> rsd;
  bits<5> rsdr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rsdr = $rsd";
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rsd{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11-7} = rs0{4-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 0;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rsdr = $rsd" in 
def ESP_SAT_P : PseudoESP32P4<(outs GPRPIE:$rsdr), (ins GPR:$rs0, GPR:$rs1, GPR:$rsd),
				"!esp_sat_p $rsd, $rs0, $rs1",
			[(set GPRPIE:$rsdr, (int_riscv_esp_sat GPR:$rs0, GPR:$rs1, GPR:$rsd))]>;

def ESP_SUBX2: Esp32P4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.subx2\t $rd, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 0;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SUBX2_P : PseudoESP32P4<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rd),
				"!esp_subx2_p $rd, $rs1, $rs2",
			[(int_riscv_esp_subx2 GPR:$rs1, GPR:$rs2, GPR:$rd)]>;

def ESP_SUBX4: Esp32P4Inst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   "esp.subx4\t $rd, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
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
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11-7} = rd{4-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 0;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SUBX4_P : PseudoESP32P4<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rd),
				"!esp_subx4_p $rd, $rs1, $rs2",
			[(int_riscv_esp_subx4 GPR:$rs1, GPR:$rs2, GPR:$rd)]>;

def ESP_ANDQ: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.andq\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_ANDQ_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_andq_p $qz, $qx, $qy",
			[(int_riscv_esp_andq timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_NOTQ: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx),
                   "esp.notq\t $qz, $qx", []> 
{
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_NOTQ_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qz),
				"!esp_notq_p $qz, $qx",
			[(int_riscv_esp_notq timm:$qx, timm:$qz)]>;

def ESP_ORQ: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.orq\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_ORQ_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_orq_p $qz, $qx, $qy",
			[(int_riscv_esp_orq timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_XORQ: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.xorq\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_XORQ_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_xorq_p $qz, $qx, $qy",
			[(int_riscv_esp_xorq timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_S16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.s16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_S32: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.s32\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_S32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_s32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_s32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_S8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.s8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_S8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_U16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.u16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_U16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_U32: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.u32\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_U32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_u32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_u32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_U8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.u8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_U8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_u8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_S16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.s16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_S32: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.s32\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_S32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_s32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_s32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_S8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.s8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_S8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_U16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.u16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_U16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_U32: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.u32\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_U32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_u32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_u32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_U8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.u8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_U8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_u8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_S16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.s16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_S32: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.s32\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_S32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_s32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_s32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_S8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.s8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_S8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_U16: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.u16\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_U16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_U32: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.u32\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_U32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_u32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_u32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_U8: Esp32P4Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.u8\t $qz, $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_U8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_u8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_MOV_S16_QACC: Esp32P4Inst<(outs), (ins QR:$qu),
                   "esp.mov.s16.qacc\t $qu", []> 
{
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
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

let usesCustomInserter = 1 in 
def ESP_MOV_S16_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qu),
				"!esp_mov_s16_qacc_p $qu",
			[(int_riscv_esp_mov_s16_qacc timm:$qu)]>;

def ESP_MOV_S8_QACC: Esp32P4Inst<(outs), (ins QR:$qu),
                   "esp.mov.s8.qacc\t $qu", []> 
{
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
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

let usesCustomInserter = 1 in 
def ESP_MOV_S8_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qu),
				"!esp_mov_s8_qacc_p $qu",
			[(int_riscv_esp_mov_s8_qacc timm:$qu)]>;

def ESP_MOV_U16_QACC: Esp32P4Inst<(outs), (ins QR:$qu),
                   "esp.mov.u16.qacc\t $qu", []> 
{
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
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

let usesCustomInserter = 1 in 
def ESP_MOV_U16_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qu),
				"!esp_mov_u16_qacc_p $qu",
			[(int_riscv_esp_mov_u16_qacc timm:$qu)]>;

def ESP_MOV_U8_QACC: Esp32P4Inst<(outs), (ins QR:$qu),
                   "esp.mov.u8.qacc\t $qu", []> 
{
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
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

let usesCustomInserter = 1 in 
def ESP_MOV_U8_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qu),
				"!esp_mov_u8_qacc_p $qu",
			[(int_riscv_esp_mov_u8_qacc timm:$qu)]>;

def ESP_MOVI_16_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qy, select_16:$sel16),
                   "esp.movi.16.a\t $qy, $rd, $sel16", []> 
{
  bits<3> qy;
  bits<4> sel16;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_16_A_P : PseudoESP32P4<(outs), (ins imm8:$qy, select_16:$sel16, GPRPIE:$rd),
				"!esp_movi_16_a_p $qy, $rd, $sel16",
			[(int_riscv_esp_movi_16_a timm:$qy, timm:$sel16, GPRPIE:$rd)]>;

def ESP_MOVI_16_Q: Esp32P4Inst<(outs QR:$qy), (ins GPRPIE:$rs1, select_16:$sel16),
                   "esp.movi.16.q\t $qy, $rs1, $sel16", []> 
{
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10-7} = sel16{3-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_16_Q_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, select_16:$sel16, imm8:$qy),
				"!esp_movi_16_q_p $qy, $rs1, $sel16",
			[(int_riscv_esp_movi_16_q GPRPIE:$rs1, timm:$sel16, timm:$qy)]>;

def ESP_MOVI_32_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qy, select_4:$sel4),
                   "esp.movi.32.a\t $qy, $rd, $sel4", []> 
{
  bits<3> qy;
  bits<2> sel4;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = sel4{1};
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = sel4{0};
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_32_A_P : PseudoESP32P4<(outs), (ins imm8:$qy, select_4:$sel4, GPRPIE:$rd),
				"!esp_movi_32_a_p $qy, $rd, $sel4",
			[(int_riscv_esp_movi_32_a timm:$qy, timm:$sel4, GPRPIE:$rd)]>;

def ESP_MOVI_32_Q: Esp32P4Inst<(outs QR:$qy), (ins GPRPIE:$rs1, select_4:$sel4),
                   "esp.movi.32.q\t $qy, $rs1, $sel4", []> 
{
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10-9} = sel4{1-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_32_Q_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, select_4:$sel4, imm8:$qy),
				"!esp_movi_32_q_p $qy, $rs1, $sel4",
			[(int_riscv_esp_movi_32_q GPRPIE:$rs1, timm:$sel4, timm:$qy)]>;

def ESP_MOVI_8_A: Esp32P4Inst<(outs GPRPIE:$rd), (ins QR:$qy, select_16:$sel16),
                   "esp.movi.8.a\t $qy, $rd, $sel16", []> 
{
  bits<3> qy;
  bits<4> sel16;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_8_A_P : PseudoESP32P4<(outs), (ins imm8:$qy, select_16:$sel16, GPRPIE:$rd),
				"!esp_movi_8_a_p $qy, $rd, $sel16",
			[(int_riscv_esp_movi_8_a timm:$qy, timm:$sel16, GPRPIE:$rd)]>;

def ESP_MOVI_8_Q: Esp32P4Inst<(outs QR:$qy), (ins GPRPIE:$rs1, select_16:$sel16),
                   "esp.movi.8.q\t $qy, $rs1, $sel16", []> 
{
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10-7} = sel16{3-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_8_Q_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, select_16:$sel16, imm8:$qy),
				"!esp_movi_8_q_p $qy, $rs1, $sel16",
			[(int_riscv_esp_movi_8_q GPRPIE:$rs1, timm:$sel16, timm:$qy)]>;

def ESP_MOVX_R_CFG: Esp32P4Inst<(outs GPRPIE:$rd), (ins),
                   "esp.movx.r.cfg\t $rd", []> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_CFG_P : PseudoESP32P4<(outs GPRPIE:$rd), (ins),
				"!esp_movx_r_cfg_p $rd",
			[(set GPRPIE:$rd, (int_riscv_esp_movx_r_cfg))]>;

def ESP_MOVX_R_FFT_BIT_WIDTH: Esp32P4Inst<(outs GPRPIE:$rd), (ins),
                   "esp.movx.r.fft.bit.width\t $rd", []> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_FFT_BIT_WIDTH_P : PseudoESP32P4<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_fft_bit_width_p $rd",
			[(int_riscv_esp_movx_r_fft_bit_width GPRPIE:$rd)]>;

def ESP_MOVX_R_PERF: Esp32P4Inst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1),
                   "esp.movx.r.perf\t $rd, $rs1", []> 
{
  bits<5> rs1;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_PERF_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rd),
				"!esp_movx_r_perf_p $rd, $rs1",
			[(int_riscv_esp_movx_r_perf GPRPIE:$rs1, GPRPIE:$rd)]>;

def ESP_MOVX_R_SAR: Esp32P4Inst<(outs GPRPIE:$rd), (ins),
                   "esp.movx.r.sar\t $rd", []> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_SAR_P : PseudoESP32P4<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_sar_p $rd",
			[(int_riscv_esp_movx_r_sar GPRPIE:$rd)]>;

def ESP_MOVX_R_SAR_BYTES: Esp32P4Inst<(outs GPRPIE:$rd), (ins),
                   "esp.movx.r.sar.bytes\t $rd", []> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_SAR_BYTES_P : PseudoESP32P4<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_sar_bytes_p $rd",
			[(int_riscv_esp_movx_r_sar_bytes GPRPIE:$rd)]>;

def ESP_MOVX_R_XACC_H: Esp32P4Inst<(outs GPRPIE:$rd), (ins),
                   "esp.movx.r.xacc.h\t $rd", []> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_XACC_H_P : PseudoESP32P4<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_xacc_h_p $rd",
			[(int_riscv_esp_movx_r_xacc_h GPRPIE:$rd)]>;

def ESP_MOVX_R_XACC_L: Esp32P4Inst<(outs GPRPIE:$rd), (ins),
                   "esp.movx.r.xacc.l\t $rd", []> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_XACC_L_P : PseudoESP32P4<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_xacc_l_p $rd",
			[(int_riscv_esp_movx_r_xacc_l GPRPIE:$rd)]>;

def ESP_MOVX_W_CFG: Esp32P4Inst<(outs), (ins GPRPIE:$rs1),
                   "esp.movx.w.cfg\t $rs1", []> 
{
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_CFG_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_cfg_p $rs1",
			[(int_riscv_esp_movx_w_cfg GPRPIE:$rs1)]>;

def ESP_MOVX_W_FFT_BIT_WIDTH: Esp32P4Inst<(outs), (ins GPRPIE:$rs1),
                   "esp.movx.w.fft.bit.width\t $rs1", []> 
{
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_FFT_BIT_WIDTH_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_fft_bit_width_p $rs1",
			[(int_riscv_esp_movx_w_fft_bit_width GPRPIE:$rs1)]>;

def ESP_MOVX_W_PERF: Esp32P4Inst<(outs), (ins GPRPIE:$rs1),
                   "esp.movx.w.perf\t $rs1", []> 
{
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_PERF_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_perf_p $rs1",
			[(int_riscv_esp_movx_w_perf GPRPIE:$rs1)]>;

def ESP_MOVX_W_SAR: Esp32P4Inst<(outs), (ins GPRPIE:$rs1),
                   "esp.movx.w.sar\t $rs1", []> 
{
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_SAR_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_sar_p $rs1",
			[(int_riscv_esp_movx_w_sar GPRPIE:$rs1)]>;

def ESP_MOVX_W_SAR_BYTES: Esp32P4Inst<(outs), (ins GPRPIE:$rs1),
                   "esp.movx.w.sar.bytes\t $rs1", []> 
{
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_SAR_BYTES_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_sar_bytes_p $rs1",
			[(int_riscv_esp_movx_w_sar_bytes GPRPIE:$rs1)]>;

def ESP_MOVX_W_XACC_H: Esp32P4Inst<(outs), (ins GPRPIE:$rs1),
                   "esp.movx.w.xacc.h\t $rs1", []> 
{
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_XACC_H_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_xacc_h_p $rs1",
			[(int_riscv_esp_movx_w_xacc_h GPRPIE:$rs1)]>;

def ESP_MOVX_W_XACC_L: Esp32P4Inst<(outs), (ins GPRPIE:$rs1),
                   "esp.movx.w.xacc.l\t $rs1", []> 
{
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_XACC_L_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_xacc_l_p $rs1",
			[(int_riscv_esp_movx_w_xacc_l GPRPIE:$rs1)]>;

def ESP_VEXT_S16: Esp32P4Inst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
                   "esp.vext.s16\t $qz, $qv, $qw", []> 
{
  bits<3> qw;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VEXT_S16_P : PseudoESP32P4<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_s16_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_s16 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VEXT_S8: Esp32P4Inst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
                   "esp.vext.s8\t $qz, $qv, $qw", []> 
{
  bits<3> qw;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VEXT_S8_P : PseudoESP32P4<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_s8_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_s8 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VEXT_U16: Esp32P4Inst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
                   "esp.vext.u16\t $qz, $qv, $qw", []> 
{
  bits<3> qw;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VEXT_U16_P : PseudoESP32P4<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_u16_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_u16 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VEXT_U8: Esp32P4Inst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
                   "esp.vext.u8\t $qz, $qv, $qw", []> 
{
  bits<3> qw;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VEXT_U8_P : PseudoESP32P4<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_u8_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_u8 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VUNZIP_16: Esp32P4Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vunzip.16\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qxr = $qx, $qyr = $qy";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VUNZIP_16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vunzip_16_p $qx, $qy",
			[(int_riscv_esp_vunzip_16 timm:$qx, timm:$qy)]>;

def ESP_VUNZIP_32: Esp32P4Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vunzip.32\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qxr = $qx, $qyr = $qy";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 1;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VUNZIP_32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vunzip_32_p $qx, $qy",
			[(int_riscv_esp_vunzip_32 timm:$qx, timm:$qy)]>;

def ESP_VUNZIP_8: Esp32P4Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vunzip.8\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qxr = $qx, $qyr = $qy";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VUNZIP_8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vunzip_8_p $qx, $qy",
			[(int_riscv_esp_vunzip_8 timm:$qx, timm:$qy)]>;

def ESP_VUNZIPT_16: Esp32P4Inst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
                   "esp.vunzipt.16\t $qx, $qy, $qw", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<3> qxr;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qxr = $qx, $qyr = $qy, $qwr = $qw";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
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

let usesCustomInserter = 1 in 
def ESP_VUNZIPT_16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vunzipt_16_p $qx, $qy, $qw",
			[(int_riscv_esp_vunzipt_16 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_VUNZIPT_8: Esp32P4Inst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
                   "esp.vunzipt.8\t $qx, $qy, $qw", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<3> qxr;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qxr = $qx, $qyr = $qy, $qwr = $qw";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
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

let usesCustomInserter = 1 in 
def ESP_VUNZIPT_8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vunzipt_8_p $qx, $qy, $qw",
			[(int_riscv_esp_vunzipt_8 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_VZIP_16: Esp32P4Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vzip.16\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qxr = $qx, $qyr = $qy";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VZIP_16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vzip_16_p $qx, $qy",
			[(int_riscv_esp_vzip_16 timm:$qx, timm:$qy)]>;

def ESP_VZIP_32: Esp32P4Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vzip.32\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qxr = $qx, $qyr = $qy";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 1;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VZIP_32_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vzip_32_p $qx, $qy",
			[(int_riscv_esp_vzip_32 timm:$qx, timm:$qy)]>;

def ESP_VZIP_8: Esp32P4Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vzip.8\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qxr = $qx, $qyr = $qy";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VZIP_8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vzip_8_p $qx, $qy",
			[(int_riscv_esp_vzip_8 timm:$qx, timm:$qy)]>;

def ESP_VZIPT_16: Esp32P4Inst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
                   "esp.vzipt.16\t $qx, $qy, $qw", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<3> qxr;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qxr = $qx, $qyr = $qy, $qwr = $qw";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
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

let usesCustomInserter = 1 in 
def ESP_VZIPT_16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vzipt_16_p $qx, $qy, $qw",
			[(int_riscv_esp_vzipt_16 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_VZIPT_8: Esp32P4Inst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
                   "esp.vzipt.8\t $qx, $qy, $qw", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<3> qxr;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qxr = $qx, $qyr = $qy, $qwr = $qw";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 1;
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

let usesCustomInserter = 1 in 
def ESP_VZIPT_8_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vzipt_8_p $qx, $qy, $qw",
			[(int_riscv_esp_vzipt_8 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_ZERO_Q: Esp32P4Inst<(outs QR:$qz), (ins),
                   "esp.zero.q\t $qz", []> 
{
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_ZERO_Q_P : PseudoESP32P4<(outs), (ins imm8:$qz),
				"!esp_zero_q_p $qz",
			[(int_riscv_esp_zero_q timm:$qz)]>;

def ESP_ZERO_QACC: Esp32P4Inst<(outs), (ins),
                   "esp.zero.qacc\t", []> 
{
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
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

let usesCustomInserter = 1 in 
def ESP_ZERO_QACC_P : PseudoESP32P4<(outs), (ins),
				"!esp_zero_qacc_p",
			[(int_riscv_esp_zero_qacc)]>;

def ESP_ZERO_XACC: Esp32P4Inst<(outs), (ins),
                   "esp.zero.xacc\t", []> 
{
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
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

let usesCustomInserter = 1 in 
def ESP_ZERO_XACC_P : PseudoESP32P4<(outs), (ins),
				"!esp_zero_xacc_p",
			[(int_riscv_esp_zero_xacc)]>;

def ESP_FFT_AMS_S16_LD_INCP: Esp32P4Inst<(outs QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2),
                   "esp.fft.ams.s16.ld.incp\t $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<5> rs1;
  bits<1> sel2;
  bits<3> qu;
  bits<3> qz;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = sel2{0};
  let Inst{22-20} = qv{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_AMS_S16_LD_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qw, GPRPIE:$rs1, select_2:$sel2, imm8:$qu, imm8:$qz, imm8:$qv),
				"!esp_fft_ams_s16_ld_incp_p $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_ams_s16_ld_incp timm:$qx, timm:$qy, timm:$qw, GPRPIE:$rs1, timm:$sel2, timm:$qu, timm:$qz, timm:$qv))]>;

def ESP_FFT_AMS_S16_LD_INCP_UAUP: Esp32P4Inst<(outs QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2),
                   "esp.fft.ams.s16.ld.incp.uaup\t $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<5> rs1;
  bits<1> sel2;
  bits<3> qu;
  bits<3> qz;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = sel2{0};
  let Inst{22-20} = qv{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_AMS_S16_LD_INCP_UAUP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qw, GPRPIE:$rs1, select_2:$sel2, imm8:$qu, imm8:$qz, imm8:$qv),
				"!esp_fft_ams_s16_ld_incp_uaup_p $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_ams_s16_ld_incp_uaup timm:$qx, timm:$qy, timm:$qw, GPRPIE:$rs1, timm:$sel2, timm:$qu, timm:$qz, timm:$qv))]>;

def ESP_FFT_AMS_S16_LD_R32_DECP: Esp32P4Inst<(outs QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2),
                   "esp.fft.ams.s16.ld.r32.decp\t $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<5> rs1;
  bits<1> sel2;
  bits<3> qu;
  bits<3> qz;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = sel2{0};
  let Inst{22-20} = qv{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_AMS_S16_LD_R32_DECP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qw, GPRPIE:$rs1, select_2:$sel2, imm8:$qu, imm8:$qz, imm8:$qv),
				"!esp_fft_ams_s16_ld_r32_decp_p $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_ams_s16_ld_r32_decp timm:$qx, timm:$qy, timm:$qw, GPRPIE:$rs1, timm:$sel2, timm:$qu, timm:$qz, timm:$qv))]>;

def ESP_FFT_AMS_S16_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r, GPRPIE:$rs2r), (ins QR:$qx, QR:$qy, QR:$qw, QR:$qu, GPRPIE:$rs1, GPRPIE:$rs2, select_2:$sel2),
                   "esp.fft.ams.s16.st.incp\t $qu, $qz, $rs2, $rs1, $qx, $qw, $qy, $sel2", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs2;
  bits<1> sel2;
  bits<3> qz;
  bits<5> rs1r;
  bits<5> rs2r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1, $rs2r = $rs2";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = sel2{0};
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_FFT_AMS_S16_ST_INCP_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw, imm8:$qu, GPRPIE:$rs1, GPRPIE:$rs2, select_2:$sel2, imm8:$qz),
				"!esp_fft_ams_s16_st_incp_p $qu, $qz, $rs2, $rs1, $qx, $qw, $qy, $sel2",
			[(int_riscv_esp_fft_ams_s16_st_incp timm:$qx, timm:$qy, timm:$qw, timm:$qu, GPRPIE:$rs1, GPRPIE:$rs2, timm:$sel2, timm:$qz)]>;

def ESP_FFT_BITREV: Esp32P4Inst<(outs GPRPIE:$rs1r, QR:$qvr), (ins GPRPIE:$rs1, QR:$qv),
                   "esp.fft.bitrev\t $qv, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  bits<3> qvr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1, $qvr = $qv";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
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

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_BITREV_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, imm8:$qv),
				"!esp_fft_bitrev_p $qv, $rs1",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_bitrev GPRPIE:$rs1, timm:$qv))]>;

def ESP_FFT_CMUL_S16_LD_XP: Esp32P4Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, select_8:$sel8),
                   "esp.fft.cmul.s16.ld.xp\t $qu, $rs1, $rs2, $qz, $qy, $qx, $sel8", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> sel8;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = sel8{2-1};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = sel8{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_CMUL_S16_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_8:$sel8, imm8:$qz, imm8:$qu),
				"!esp_fft_cmul_s16_ld_xp_p $qu, $rs1, $rs2, $qz, $qy, $qx, $sel8",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_cmul_s16_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel8, timm:$qz, timm:$qu))]>;

def ESP_FFT_CMUL_S16_ST_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4, select_4:$upd4, select_8:$sel8),
                   "esp.fft.cmul.s16.st.xp\t $qy, $qx, $qu, $rs1, $rs2, $sel8, $upd4, $sel4", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<2> sel4;
  bits<2> upd4;
  bits<3> sel8;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = sel4{1-0};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = upd4{1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = upd4{0};
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = sel8{2-0};
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_CMUL_S16_ST_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, select_4:$sel4, select_4:$upd4, select_8:$sel8),
				"!esp_fft_cmul_s16_st_xp_p $qy, $qx, $qu, $rs1, $rs2, $sel8, $upd4, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_cmul_s16_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sel4, timm:$upd4, timm:$sel8))]>;

def ESP_FFT_R2BF_S16: Esp32P4Inst<(outs QR:$qz, QR:$qv), (ins QR:$qx, QR:$qy, select_2:$sel2),
                   "esp.fft.r2bf.s16\t $qz, $qv, $qx, $qy, $sel2", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sel2;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = sel2{0};
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_FFT_R2BF_S16_P : PseudoESP32P4<(outs), (ins imm8:$qx, imm8:$qy, select_2:$sel2, imm8:$qz, imm8:$qv),
				"!esp_fft_r2bf_s16_p $qz, $qv, $qx, $qy, $sel2",
			[(int_riscv_esp_fft_r2bf_s16 timm:$qx, timm:$qy, timm:$sel2, timm:$qz, timm:$qv)]>;

def ESP_FFT_R2BF_S16_ST_INCP: Esp32P4Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4),
                   "esp.fft.r2bf.s16.st.incp\t $qz, $qx, $qy, $rs1, $sel4", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23-22} = sel4{1-0};
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_R2BF_S16_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_4:$sel4, imm8:$qz),
				"!esp_fft_r2bf_s16_st_incp_p $qz, $qx, $qy, $rs1, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_r2bf_s16_st_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel4, timm:$qz))]>;

def ESP_FFT_VST_R32_DECP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, select_2:$sel2),
                   "esp.fft.vst.r32.decp\t $qu, $rs1, $sel2", []> 
{
  bits<3> qu;
  bits<5> rs1;
  bits<1> sel2;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = sel2{0};
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_VST_R32_DECP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, select_2:$sel2),
				"!esp_fft_vst_r32_decp_p $qu, $rs1, $sel2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_vst_r32_decp timm:$qu, GPRPIE:$rs1, timm:$sel2))]>;

def ESP_LD_128_USAR_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.128.usar.ip\t $qu, $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = off25616{7-5};
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{4-1};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off25616{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_128_USAR_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616, imm8:$qu),
				"!esp_ld_128_usar_ip_p $qu, $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_128_usar_ip GPRPIE:$rs1, timm:$off25616, timm:$qu))]>;

def ESP_LD_128_USAR_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ld.128.usar.xp\t $qu, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_128_USAR_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_ld_128_usar_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_128_usar_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_LD_XACC_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.ld.xacc.ip\t $rs1, $off2568", []> 
{
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = off2568{7};
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off2568{6-3};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = off2568{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_XACC_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_ld_xacc_ip_p $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_xacc_ip GPRPIE:$rs1, timm:$off2568))]>;

def ESP_LDQA_S16_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ldqa.s16.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_S16_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_s16_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s16_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDQA_S16_128_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ldqa.s16.128.xp\t $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_S16_128_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_s16_128_xp_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s16_128_xp GPRPIE:$rs2, GPRPIE:$rs1))]>;

def ESP_LDQA_S8_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ldqa.s8.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_S8_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_s8_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s8_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDQA_S8_128_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ldqa.s8.128.xp\t $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_S8_128_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_s8_128_xp_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s8_128_xp GPRPIE:$rs2, GPRPIE:$rs1))]>;

def ESP_LDQA_U16_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ldqa.u16.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_U16_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_u16_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u16_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDQA_U16_128_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ldqa.u16.128.xp\t $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
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

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_U16_128_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_u16_128_xp_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u16_128_xp GPRPIE:$rs2, GPRPIE:$rs1))]>;

def ESP_LDQA_U8_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ldqa.u8.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_U8_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_u8_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u8_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDQA_U8_128_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ldqa.u8.128.xp\t $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
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

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_U8_128_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_u8_128_xp_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u8_128_xp GPRPIE:$rs2, GPRPIE:$rs1))]>;

def ESP_VLDBC_16_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_4:$off2564),
                   "esp.vldbc.16.ip\t $qu, $rs1, $off2564", []> 
{
  bits<5> rs1;
  bits<8> off2564;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30-28} = off2564{7-5};
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23-20} = off2564{4-1};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off2564{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_16_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_4:$off2564, imm8:$qu),
				"!esp_vldbc_16_ip_p $qu, $rs1, $off2564",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_16_ip GPRPIE:$rs1, timm:$off2564, timm:$qu))]>;

def ESP_VLDBC_16_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldbc.16.xp\t $qu, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_16_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vldbc_16_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_16_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLDBC_32_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_4:$off2564),
                   "esp.vldbc.32.ip\t $qu, $rs1, $off2564", []> 
{
  bits<5> rs1;
  bits<8> off2564;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-29} = off2564{7-5};
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23-20} = off2564{4-1};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off2564{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_32_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_4:$off2564, imm8:$qu),
				"!esp_vldbc_32_ip_p $qu, $rs1, $off2564",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_32_ip GPRPIE:$rs1, timm:$off2564, timm:$qu))]>;

def ESP_VLDBC_32_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldbc.32.xp\t $qu, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_32_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vldbc_32_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_32_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLDBC_8_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_4:$off2564),
                   "esp.vldbc.8.ip\t $qu, $rs1, $off2564", []> 
{
  bits<5> rs1;
  bits<8> off2564;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30-28} = off2564{7-5};
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23-20} = off2564{4-1};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off2564{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_8_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_4:$off2564, imm8:$qu),
				"!esp_vldbc_8_ip_p $qu, $rs1, $off2564",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_8_ip GPRPIE:$rs1, timm:$off2564, timm:$qu))]>;

def ESP_VLDBC_8_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldbc.8.xp\t $qu, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_8_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vldbc_8_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_8_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLDEXT_S16_IP: Esp32P4Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vldext.s16.ip\t $qu, $qz, $rs1, $off1616", []> 
{
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off1616{3-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_S16_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_s16_ip_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s16_ip GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_S16_XP: Esp32P4Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldext.s16.xp\t $qu, $qz, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_S16_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_s16_xp_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s16_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_S8_IP: Esp32P4Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vldext.s8.ip\t $qu, $qz, $rs1, $off1616", []> 
{
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off1616{3-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_S8_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_s8_ip_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s8_ip GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_S8_XP: Esp32P4Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldext.s8.xp\t $qu, $qz, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_S8_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_s8_xp_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s8_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U16_IP: Esp32P4Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vldext.u16.ip\t $qu, $qz, $rs1, $off1616", []> 
{
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off1616{3-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_U16_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_u16_ip_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u16_ip GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U16_XP: Esp32P4Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldext.u16.xp\t $qu, $qz, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_U16_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_u16_xp_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u16_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U8_IP: Esp32P4Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vldext.u8.ip\t $qu, $qz, $rs1, $off1616", []> 
{
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off1616{3-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_U8_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_u8_ip_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u8_ip GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U8_XP: Esp32P4Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldext.u8.xp\t $qu, $qz, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_U8_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_u8_xp_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u8_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDHBC_16_INCP: Esp32P4Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1),
                   "esp.vldhbc.16.incp\t $qu, $qz, $rs1", []> 
{
  bits<5> rs1;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDHBC_16_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldhbc_16_incp_p $qu, $qz, $rs1",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldhbc_16_incp GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_LD_QACC_H_H_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.qacc.h.h.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_QACC_H_H_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_h_h_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_h_h_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LD_QACC_H_L_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.qacc.h.l.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_QACC_H_L_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_h_l_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_h_l_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LD_QACC_L_H_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.qacc.l.h.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_QACC_L_H_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_l_h_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_l_h_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LD_QACC_L_L_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.qacc.l.l.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_QACC_L_L_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_l_l_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_l_l_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LD_UA_STATE_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.ua.state.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = off25616{7};
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{6-3};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = off25616{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_UA_STATE_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_ua_state_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_ua_state_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDXQ_32: Esp32P4Inst<(outs QR:$qu), (ins GPRPIE:$rs1, QR:$qw, select_4:$sel4, select_8:$sel8),
                   "esp.ldxq.32\t $qu, $qw, $rs1, $sel4, $sel8", []> 
{
  bits<5> rs1;
  bits<3> qw;
  bits<2> sel4;
  bits<3> sel8;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30-29} = sel4{1-0};
  let Inst{28-26} = sel8{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_LDXQ_32_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qw, select_4:$sel4, select_8:$sel8, imm8:$qu),
				"!esp_ldxq_32_p $qu, $qw, $rs1, $sel4, $sel8",
			[(int_riscv_esp_ldxq_32 GPRPIE:$rs1, timm:$qw, timm:$sel4, timm:$sel8, timm:$qu)]>;

def ESP_ST_QACC_H_H_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.h.h.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_QACC_H_H_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_h_h_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_h_h_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_ST_QACC_H_L_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.h.l.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_QACC_H_L_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_h_l_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_h_l_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_ST_QACC_L_H_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.l.h.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_QACC_L_H_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_l_h_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_l_h_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_ST_QACC_L_L_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.l.l.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{7-4};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_QACC_L_L_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_l_l_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_l_l_128_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_ST_UA_STATE_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.ua.state.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = off25616{7};
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{6-3};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = off25616{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_UA_STATE_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_ua_state_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_ua_state_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_STXQ_32: Esp32P4Inst<(outs), (ins GPRPIE:$rs1, QR:$qw, QR:$qu, select_4:$sel4, select_8:$sel8),
                   "esp.stxq.32\t $qu, $qw, $rs1, $sel4, $sel8", []> 
{
  bits<5> rs1;
  bits<3> qw;
  bits<3> qu;
  bits<2> sel4;
  bits<3> sel8;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30-29} = sel4{1-0};
  let Inst{28-26} = sel8{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_STXQ_32_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, imm8:$qw, imm8:$qu, select_4:$sel4, select_8:$sel8),
				"!esp_stxq_32_p $qu, $qw, $rs1, $sel4, $sel8",
			[(int_riscv_esp_stxq_32 GPRPIE:$rs1, timm:$qw, timm:$qu, timm:$sel4, timm:$sel8)]>;

def ESP_VLD_128_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.vld.128.ip\t $qu, $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30-28} = off25616{7-5};
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{4-1};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off25616{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616, imm8:$qu),
				"!esp_vld_128_ip_p $qu, $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_128_ip GPRPIE:$rs1, timm:$off25616, timm:$qu))]>;

def ESP_VLD_128_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vld.128.xp\t $qu, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_128_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vld_128_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_128_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLD_H_64_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.vld.h.64.ip\t $qu, $rs1, $off2568", []> 
{
  bits<5> rs1;
  bits<8> off2568;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29-27} = off2568{7-5};
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off2568{4-1};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off2568{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_H_64_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568, imm8:$qu),
				"!esp_vld_h_64_ip_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_h_64_ip GPRPIE:$rs1, timm:$off2568, timm:$qu))]>;

def ESP_VLD_H_64_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vld.h.64.xp\t $qu, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_H_64_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vld_h_64_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_h_64_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLD_L_64_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.vld.l.64.ip\t $qu, $rs1, $off2568", []> 
{
  bits<5> rs1;
  bits<8> off2568;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29-27} = off2568{7-5};
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off2568{4-1};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off2568{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_L_64_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568, imm8:$qu),
				"!esp_vld_l_64_ip_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_l_64_ip GPRPIE:$rs1, timm:$off2568, timm:$qu))]>;

def ESP_VLD_L_64_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vld.l.64.xp\t $qu, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_L_64_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vld_l_64_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_l_64_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VST_128_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.vst.128.ip\t $qu, $rs1, $off25616", []> 
{
  bits<3> qu;
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30-28} = off25616{7-5};
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23-20} = off25616{4-1};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off25616{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_128_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_vst_128_ip_p $qu, $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_128_ip timm:$qu, GPRPIE:$rs1, timm:$off25616))]>;

def ESP_VST_128_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qu, GPRPIE:$rs1),
                   "esp.vst.128.xp\t $qu, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_128_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qu, GPRPIE:$rs1),
				"!esp_vst_128_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_128_xp GPRPIE:$rs2, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VST_H_64_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.vst.h.64.ip\t $qu, $rs1, $off2568", []> 
{
  bits<3> qu;
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29-27} = off2568{7-5};
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off2568{4-1};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off2568{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_H_64_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_vst_h_64_ip_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_h_64_ip timm:$qu, GPRPIE:$rs1, timm:$off2568))]>;

def ESP_VST_H_64_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qu, GPRPIE:$rs1),
                   "esp.vst.h.64.xp\t $qu, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_H_64_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qu, GPRPIE:$rs1),
				"!esp_vst_h_64_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_h_64_xp GPRPIE:$rs2, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VST_L_64_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.vst.l.64.ip\t $qu, $rs1, $off2568", []> 
{
  bits<3> qu;
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29-27} = off2568{7-5};
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off2568{4-1};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off2568{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_L_64_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_vst_l_64_ip_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_l_64_ip timm:$qu, GPRPIE:$rs1, timm:$off2568))]>;

def ESP_VST_L_64_XP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qu, GPRPIE:$rs1),
                   "esp.vst.l.64.xp\t $qu, $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_L_64_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qu, GPRPIE:$rs1),
				"!esp_vst_l_64_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_l_64_xp GPRPIE:$rs2, timm:$qu, GPRPIE:$rs1))]>;

def ESP_SLCI_2Q: Esp32P4Inst<(outs QR:$qyr, QR:$qwr), (ins QR:$qy, QR:$qw, select_16:$sel16),
                   "esp.slci.2q\t $qy, $qw, $sel16", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<4> sel16;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qyr = $qy, $qwr = $qw";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = sel16{3};
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = sel16{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SLCI_2Q_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qw, select_16:$sel16),
				"!esp_slci_2q_p $qy, $qw, $sel16",
			[(int_riscv_esp_slci_2q timm:$qy, timm:$qw, timm:$sel16)]>;

def ESP_SLCXXP_2Q: Esp32P4Inst<(outs QR:$qyr, QR:$qwr), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy, QR:$qw),
                   "esp.slcxxp.2q\t $qy, $qw, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qy;
  bits<3> qw;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qyr = $qy, $qwr = $qw";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
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
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SLCXXP_2Q_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy, imm8:$qw),
				"!esp_slcxxp_2q_p $qy, $qw, $rs1, $rs2",
			[(int_riscv_esp_slcxxp_2q GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy, timm:$qw)]>;

def ESP_SRC_Q: Esp32P4Inst<(outs QR:$qz), (ins QR:$qy, QR:$qw),
                   "esp.src.q\t $qz, $qw, $qy", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRC_Q_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qz),
				"!esp_src_q_p $qz, $qw, $qy",
			[(int_riscv_esp_src_q timm:$qy, timm:$qw, timm:$qz)]>;

def ESP_SRC_Q_LD_IP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r, QR:$qwr), (ins QR:$qy, GPRPIE:$rs1, QR:$qw, offset_256_16:$off25616),
                   "esp.src.q.ld.ip\t $qu, $rs1, $off25616, $qw, $qy", []> 
{
  bits<3> qy;
  bits<5> rs1;
  bits<3> qw;
  bits<8> off25616;
  bits<3> qu;
  bits<5> rs1r;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1, $qwr = $qw";
  
  let Inst{31-29} = off25616{7-5};
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23-20} = off25616{4-1};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off25616{0};
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_SRC_Q_LD_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qy, GPRPIE:$rs1, imm8:$qw, offset_256_16:$off25616, imm8:$qu),
				"!esp_src_q_ld_ip_p $qu, $rs1, $off25616, $qw, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_src_q_ld_ip timm:$qy, GPRPIE:$rs1, timm:$qw, timm:$off25616, timm:$qu))]>;

def ESP_SRC_Q_LD_XP: Esp32P4Inst<(outs QR:$qu, GPRPIE:$rs1r, QR:$qwr), (ins GPRPIE:$rs2, QR:$qy, GPRPIE:$rs1, QR:$qw),
                   "esp.src.q.ld.xp\t $qu, $rs1, $rs2, $qw, $qy", []> 
{
  bits<5> rs2;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qw;
  bits<3> qu;
  bits<5> rs1r;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1, $qwr = $qw";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_SRC_Q_LD_XP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qy, GPRPIE:$rs1, imm8:$qw, imm8:$qu),
				"!esp_src_q_ld_xp_p $qu, $rs1, $rs2, $qw, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_src_q_ld_xp GPRPIE:$rs2, timm:$qy, GPRPIE:$rs1, timm:$qw, timm:$qu))]>;

def ESP_SRC_Q_QUP: Esp32P4Inst<(outs QR:$qz, QR:$qwr), (ins QR:$qy, QR:$qw),
                   "esp.src.q.qup\t $qz, $qw, $qy", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qz;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qwr = $qw";
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRC_Q_QUP_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qz),
				"!esp_src_q_qup_p $qz, $qw, $qy",
			[(int_riscv_esp_src_q_qup timm:$qy, timm:$qw, timm:$qz)]>;

def ESP_SRCI_2Q: Esp32P4Inst<(outs QR:$qyr, QR:$qwr), (ins QR:$qy, QR:$qw, select_16:$sel16),
                   "esp.srci.2q\t $qy, $qw, $sel16", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<4> sel16;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qyr = $qy, $qwr = $qw";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = sel16{3};
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = sel16{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCI_2Q_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qw, select_16:$sel16),
				"!esp_srci_2q_p $qy, $qw, $sel16",
			[(int_riscv_esp_srci_2q timm:$qy, timm:$qw, timm:$sel16)]>;

def ESP_SRCMB_S16_Q_QACC: Esp32P4Inst<(outs QR:$qu), (ins QR:$qw, select_2:$sel2),
                   "esp.srcmb.s16.q.qacc\t $qu, $qw, $sel2", []> 
{
  bits<3> qw;
  bits<1> sel2;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = sel2{0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
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

let usesCustomInserter = 1 in 
def ESP_SRCMB_S16_Q_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qw, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_s16_q_qacc_p $qu, $qw, $sel2",
			[(int_riscv_esp_srcmb_s16_q_qacc timm:$qw, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_S16_QACC: Esp32P4Inst<(outs QR:$qu), (ins GPRPIE:$rs1, select_2:$sel2),
                   "esp.srcmb.s16.qacc\t $qu, $rs1, $sel2", []> 
{
  bits<5> rs1;
  bits<1> sel2;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = sel2{0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_S16_QACC_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_s16_qacc_p $qu, $rs1, $sel2",
			[(int_riscv_esp_srcmb_s16_qacc GPRPIE:$rs1, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_S8_Q_QACC: Esp32P4Inst<(outs QR:$qu), (ins QR:$qw, select_2:$sel2),
                   "esp.srcmb.s8.q.qacc\t $qu, $qw, $sel2", []> 
{
  bits<3> qw;
  bits<1> sel2;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = sel2{0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
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

let usesCustomInserter = 1 in 
def ESP_SRCMB_S8_Q_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qw, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_s8_q_qacc_p $qu, $qw, $sel2",
			[(int_riscv_esp_srcmb_s8_q_qacc timm:$qw, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_S8_QACC: Esp32P4Inst<(outs QR:$qu), (ins GPRPIE:$rs1, select_2:$sel2),
                   "esp.srcmb.s8.qacc\t $qu, $rs1, $sel2", []> 
{
  bits<5> rs1;
  bits<1> sel2;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = sel2{0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_S8_QACC_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_s8_qacc_p $qu, $rs1, $sel2",
			[(int_riscv_esp_srcmb_s8_qacc GPRPIE:$rs1, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_U16_Q_QACC: Esp32P4Inst<(outs QR:$qu), (ins QR:$qw, select_2:$sel2),
                   "esp.srcmb.u16.q.qacc\t $qu, $qw, $sel2", []> 
{
  bits<3> qw;
  bits<1> sel2;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = sel2{0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
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

let usesCustomInserter = 1 in 
def ESP_SRCMB_U16_Q_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qw, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_u16_q_qacc_p $qu, $qw, $sel2",
			[(int_riscv_esp_srcmb_u16_q_qacc timm:$qw, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_U16_QACC: Esp32P4Inst<(outs QR:$qu), (ins GPRPIE:$rs1, select_2:$sel2),
                   "esp.srcmb.u16.qacc\t $qu, $rs1, $sel2", []> 
{
  bits<5> rs1;
  bits<1> sel2;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = sel2{0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_U16_QACC_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_u16_qacc_p $qu, $rs1, $sel2",
			[(int_riscv_esp_srcmb_u16_qacc GPRPIE:$rs1, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_U8_Q_QACC: Esp32P4Inst<(outs QR:$qu), (ins QR:$qw, select_2:$sel2),
                   "esp.srcmb.u8.q.qacc\t $qu, $qw, $sel2", []> 
{
  bits<3> qw;
  bits<1> sel2;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = sel2{0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
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

let usesCustomInserter = 1 in 
def ESP_SRCMB_U8_Q_QACC_P : PseudoESP32P4<(outs), (ins imm8:$qw, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_u8_q_qacc_p $qu, $qw, $sel2",
			[(int_riscv_esp_srcmb_u8_q_qacc timm:$qw, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_U8_QACC: Esp32P4Inst<(outs QR:$qu), (ins GPRPIE:$rs1, select_2:$sel2),
                   "esp.srcmb.u8.qacc\t $qu, $rs1, $sel2", []> 
{
  bits<5> rs1;
  bits<1> sel2;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = sel2{0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_U8_QACC_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_u8_qacc_p $qu, $rs1, $sel2",
			[(int_riscv_esp_srcmb_u8_qacc GPRPIE:$rs1, timm:$sel2, timm:$qu)]>;

def ESP_SRCQ_128_ST_INCP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins QR:$qy, QR:$qw, GPRPIE:$rs1),
                   "esp.srcq.128.st.incp\t $qw, $qy, $rs1", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
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

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_SRCQ_128_ST_INCP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins imm8:$qy, imm8:$qw, GPRPIE:$rs1),
				"!esp_srcq_128_st_incp_p $qw, $qy, $rs1",
			[(set GPRPIE:$rs1r, (int_riscv_esp_srcq_128_st_incp timm:$qy, timm:$qw, GPRPIE:$rs1))]>;

def ESP_SRCXXP_2Q: Esp32P4Inst<(outs QR:$qyr, QR:$qwr), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy, QR:$qw),
                   "esp.srcxxp.2q\t $qy, $qw, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qy;
  bits<3> qw;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qyr = $qy, $qwr = $qw";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCXXP_2Q_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy, imm8:$qw),
				"!esp_srcxxp_2q_p $qy, $qw, $rs1, $rs2",
			[(int_riscv_esp_srcxxp_2q GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy, timm:$qw)]>;

def ESP_SRS_S_XACC: Esp32P4Inst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1),
                   "esp.srs.s.xacc\t $rd, $rs1", []> 
{
  bits<5> rs1;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRS_S_XACC_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rd),
				"!esp_srs_s_xacc_p $rd, $rs1",
			[(int_riscv_esp_srs_s_xacc GPRPIE:$rs1, GPRPIE:$rd)]>;

def ESP_SRS_U_XACC: Esp32P4Inst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1),
                   "esp.srs.u.xacc\t $rd, $rs1", []> 
{
  bits<5> rs1;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRS_U_XACC_P : PseudoESP32P4<(outs), (ins GPRPIE:$rs1, GPRPIE:$rd),
				"!esp_srs_u_xacc_p $rd, $rs1",
			[(int_riscv_esp_srs_u_xacc GPRPIE:$rs1, GPRPIE:$rd)]>;

def ESP_VSL_32: Esp32P4Inst<(outs QR:$qu), (ins QR:$qy),
                   "esp.vsl.32\t $qu, $qy", []> 
{
  bits<3> qy;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
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

let usesCustomInserter = 1 in 
def ESP_VSL_32_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qu),
				"!esp_vsl_32_p $qu, $qy",
			[(int_riscv_esp_vsl_32 timm:$qy, timm:$qu)]>;

def ESP_VSLD_16: Esp32P4Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsld.16\t $qu, $qy, $qw", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSLD_16_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsld_16_p $qu, $qy, $qw",
			[(int_riscv_esp_vsld_16 timm:$qy, timm:$qw, timm:$qu)]>;

def ESP_VSLD_32: Esp32P4Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsld.32\t $qu, $qy, $qw", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSLD_32_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsld_32_p $qu, $qy, $qw",
			[(int_riscv_esp_vsld_32 timm:$qy, timm:$qw, timm:$qu)]>;

def ESP_VSLD_8: Esp32P4Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsld.8\t $qu, $qy, $qw", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSLD_8_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsld_8_p $qu, $qy, $qw",
			[(int_riscv_esp_vsld_8 timm:$qy, timm:$qw, timm:$qu)]>;

def ESP_VSR_S32: Esp32P4Inst<(outs QR:$qu), (ins QR:$qy),
                   "esp.vsr.s32\t $qu, $qy", []> 
{
  bits<3> qy;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSR_S32_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qu),
				"!esp_vsr_s32_p $qu, $qy",
			[(int_riscv_esp_vsr_s32 timm:$qy, timm:$qu)]>;

def ESP_VSR_U32: Esp32P4Inst<(outs QR:$qu), (ins QR:$qy),
                   "esp.vsr.u32\t $qu, $qy", []> 
{
  bits<3> qy;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSR_U32_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qu),
				"!esp_vsr_u32_p $qu, $qy",
			[(int_riscv_esp_vsr_u32 timm:$qy, timm:$qu)]>;

def ESP_VSRD_16: Esp32P4Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsrd.16\t $qu, $qy, $qw", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSRD_16_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsrd_16_p $qu, $qy, $qw",
			[(int_riscv_esp_vsrd_16 timm:$qy, timm:$qw, timm:$qu)]>;

def ESP_VSRD_32: Esp32P4Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsrd.32\t $qu, $qy, $qw", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSRD_32_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsrd_32_p $qu, $qy, $qw",
			[(int_riscv_esp_vsrd_32 timm:$qy, timm:$qw, timm:$qu)]>;

def ESP_VSRD_8: Esp32P4Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsrd.8\t $qu, $qy, $qw", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSRD_8_P : PseudoESP32P4<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsrd_8_p $qu, $qy, $qw",
			[(int_riscv_esp_vsrd_8 timm:$qy, timm:$qw, timm:$qu)]>;

def ESP_ST_S_XACC_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.st.s.xacc.ip\t $rs1, $off2568", []> 
{
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = off2568{7};
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off2568{6-3};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = off2568{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_S_XACC_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_st_s_xacc_ip_p $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_s_xacc_ip GPRPIE:$rs1, timm:$off2568))]>;

def ESP_ST_U_XACC_IP: Esp32P4Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.st.u.xacc.ip\t $rs1, $off2568", []> 
{
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = off2568{7};
  let Inst{29} = 1;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-20} = off2568{6-3};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = off2568{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_U_XACC_IP_P : PseudoESP32P4<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_st_u_xacc_ip_p $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_u_xacc_ip GPRPIE:$rs1, timm:$off2568))]>;

