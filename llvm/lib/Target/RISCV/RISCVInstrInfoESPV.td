//===- RISCVInstrInfoESPV.td - RISCV Target Description -*- tablegen -*----===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISCV ESPV instructions in TableGen format.
//
// These definitions are generated
// This file is generated
//
//===----------------------------------------------------------------------===//

include "RISCVESPOperands.td"
include "RISCVInstrFormatsESP.td"
include "RISCVInstrInfoESPLOOP.td"
include "RISCVInstrInfoESPVNode.td"

// ESP_VCMULAS_S16_QACC_H: Complex multiply-accumulate to QACC_H with explicit passthru
// QACC_H_LOW:$qacc_h_l_in and QACC_H_HIGH:$qacc_h_h_in are phantom operands (not shown in assembly string)
// for data flow tracking. They are explicit operands but hidden in assembly output.
let Defs = []<Register> in
def ESP_VCMULAS_S16_QACC_H: EspVInst<(outs QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                     (ins QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy),
                   "esp.vcmulas.s16.qacc.h\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<16> qacc_h_l;      // QACC_H_LOW register (128-bit, v16i8)
  bits<16> qacc_h_h;       // QACC_H_HIGH register (128-bit, v16i8)
  bits<16> qacc_h_l_in;    // QACC_H_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_h_in;    // QACC_H_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: QACC_H input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VCMULAS_S16_QACC_H_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vcmulas_s16_qacc_h_p $qx, $qy",
			[(int_riscv_esp_vcmulas_s16_qacc_h timm:$qx, timm:$qy)]>;

def ESP_VCMULAS_S16_QACC_H_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vcmulas.s16.qacc.h.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VCMULAS_S16_QACC_H_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s16_qacc_h_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_h_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S16_QACC_H_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.vcmulas.s16.qacc.h.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qu;
  bits<5> rs1r;
  bits<16> qacc_h_l;
  bits<16> qacc_h_h;
  bits<16> qacc_h_l_in;
  bits<16> qacc_h_h_in;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VCMULAS_S16_QACC_H_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vcmulas_s16_qacc_h_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_h_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// ESP_VCMULAS_S16_QACC_L: Complex multiply-accumulate to QACC_L with explicit passthru
// QACC_L_LOW:$qacc_l_l_in and QACC_L_HIGH:$qacc_l_h_in are phantom operands (not shown in assembly string)
// for data flow tracking. They are explicit operands but hidden in assembly output.
let Defs = []<Register> in
def ESP_VCMULAS_S16_QACC_L: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h), 
                                     (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QR:$qx, QR:$qy),
                   "esp.vcmulas.s16.qacc.l\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<16> qacc_l_l;      // QACC_L_LOW register (128-bit, v16i8)
  bits<16> qacc_l_h;       // QACC_L_HIGH register (128-bit, v16i8)
  bits<16> qacc_l_l_in;    // QACC_L_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_l_h_in;    // QACC_L_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: QACC_L input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in";
  
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
def ESP_VCMULAS_S16_QACC_L_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vcmulas_s16_qacc_l_p $qx, $qy",
			[(int_riscv_esp_vcmulas_s16_qacc_l timm:$qx, timm:$qy)]>;

def ESP_VCMULAS_S16_QACC_L_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vcmulas.s16.qacc.l.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in";
  
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
def ESP_VCMULAS_S16_QACC_L_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s16_qacc_l_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_l_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S16_QACC_L_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.vcmulas.s16.qacc.l.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qu;
  bits<5> rs1r;
  bits<16> qacc_l_l;
  bits<16> qacc_l_h;
  bits<16> qacc_l_l_in;
  bits<16> qacc_l_h_in;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in";
  
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
def ESP_VCMULAS_S16_QACC_L_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vcmulas_s16_qacc_l_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_l_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// ESP_VCMULAS_S8_QACC_H: Complex multiply-accumulate to QACC_H with explicit passthru
// QACC_H_LOW:$qacc_h_l_in and QACC_H_HIGH:$qacc_h_h_in are phantom operands (not shown in assembly string)
// for data flow tracking. They are explicit operands but hidden in assembly output.
let Defs = []<Register> in
def ESP_VCMULAS_S8_QACC_H: EspVInst<(outs QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                    (ins QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy),
                   "esp.vcmulas.s8.qacc.h\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<16> qacc_h_l;      // QACC_H_LOW register (128-bit, v16i8)
  bits<16> qacc_h_h;       // QACC_H_HIGH register (128-bit, v16i8)
  bits<16> qacc_h_l_in;    // QACC_H_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_h_in;    // QACC_H_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMULAS_S8_QACC_H_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vcmulas_s8_qacc_h_p $qx, $qy",
			[(int_riscv_esp_vcmulas_s8_qacc_h timm:$qx, timm:$qy)]>;

def ESP_VCMULAS_S8_QACC_H_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vcmulas.s8.qacc.h.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  bits<16> qacc_h_l;
  bits<16> qacc_h_h;
  bits<16> qacc_h_l_in;
  bits<16> qacc_h_h_in;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VCMULAS_S8_QACC_H_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s8_qacc_h_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_h_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S8_QACC_H_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.vcmulas.s8.qacc.h.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qu;
  bits<5> rs1r;
  bits<16> qacc_h_l;
  bits<16> qacc_h_h;
  bits<16> qacc_h_l_in;
  bits<16> qacc_h_h_in;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VCMULAS_S8_QACC_H_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vcmulas_s8_qacc_h_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_h_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// ESP_VCMULAS_S8_QACC_L: Complex multiply-accumulate to QACC_L with explicit passthru
// QACC_L_LOW:$qacc_l_l_in and QACC_L_HIGH:$qacc_l_h_in are phantom operands (not shown in assembly string)
// for data flow tracking. They are explicit operands but hidden in assembly output.
let Defs = []<Register> in
def ESP_VCMULAS_S8_QACC_L: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h), 
                                    (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QR:$qx, QR:$qy),
                   "esp.vcmulas.s8.qacc.l\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<16> qacc_l_l;      // QACC_L_LOW register (128-bit, v16i8)
  bits<16> qacc_l_h;       // QACC_L_HIGH register (128-bit, v16i8)
  bits<16> qacc_l_l_in;    // QACC_L_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_l_h_in;    // QACC_L_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMULAS_S8_QACC_L_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vcmulas_s8_qacc_l_p $qx, $qy",
			[(int_riscv_esp_vcmulas_s8_qacc_l timm:$qx, timm:$qy)]>;

def ESP_VCMULAS_S8_QACC_L_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vcmulas.s8.qacc.l.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  bits<16> qacc_l_l;
  bits<16> qacc_l_h;
  bits<16> qacc_l_l_in;
  bits<16> qacc_l_h_in;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in";
  
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
def ESP_VCMULAS_S8_QACC_L_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s8_qacc_l_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_l_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S8_QACC_L_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.vcmulas.s8.qacc.l.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qu;
  bits<5> rs1r;
  bits<16> qacc_l_l;
  bits<16> qacc_l_h;
  bits<16> qacc_l_l_in;
  bits<16> qacc_l_h_in;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in";
  
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
def ESP_VCMULAS_S8_QACC_L_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vcmulas_s8_qacc_l_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_l_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// ESP_VMULAS_S16_QACC: Multiply-accumulate to QACC with explicit passthru
// QACC_H:$qacc_h_in and QACC_L:$qacc_l_in are phantom operands (not shown in assembly string)
// for data flow tracking. They are explicit operands but hidden in assembly output.
let Defs = []<Register> in
def ESP_VMULAS_S16_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                  (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy),
                   "esp.vmulas.s16.qacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<16> qacc_l_l;      // QACC_L_LOW register (128-bit, v16i8)
  bits<16> qacc_l_h;       // QACC_L_HIGH register (128-bit, v16i8)
  bits<16> qacc_h_l;       // QACC_H_LOW register (128-bit, v16i8)
  bits<16> qacc_h_h;       // QACC_H_HIGH register (128-bit, v16i8)
  bits<16> qacc_l_l_in;    // QACC_L_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_l_h_in;    // QACC_L_HIGH passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_l_in;    // QACC_H_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_h_in;    // QACC_H_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMULAS_S16_QACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_s16_qacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_s16_qacc timm:$qx, timm:$qy)]>;

def ESP_VMULAS_S16_QACC_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s16.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L and QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_S16_QACC_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s16_qacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_S16_QACC_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s16.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L and QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_S16_QACC_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s16_qacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_S16_QACC_ST_IP: EspVInst<(outs GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qu, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s16.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_S16_QACC_ST_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_s16_qacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_S16_QACC_ST_XP: EspVInst<(outs GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qu, QR:$qx, QR:$qy, GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.vmulas.s16.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_S16_QACC_ST_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_s16_qacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

// ESP.VMULAS.S16.XACC (_m version) - Multiply-accumulate with explicit state passing
// Mixed model: XACC as struct {i32 (XACC[31:0]), i8 (XACC[39:32])}
// Note: Uses manual instruction selection (RISCVISelDAGToDAG.cpp) because Pattern
// cannot handle multiple outputs directly. Pattern matching is done in RISCVInstrInfoESP32P4Patterns.td
def ESP_VMULAS_S16_XACC: EspVInst<(outs XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), 
                                   (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy),
                   "esp.vmulas.s16.xacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: XACC input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_S16_XACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_s16_xacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_s16_xacc timm:$qx, timm:$qy)]>;

// ESP.VMULAS.S16.XACC.LD.IP (_m version) - Load and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_S16_XACC_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s16.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   // Lowering code handles XACC via CopyToReg/CopyFromReg
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_S16_XACC_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s16_xacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

// ESP.VMULAS.S16.XACC.LD.XP (_m version) - Load and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_S16_XACC_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s16.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   // Lowering code handles XACC via CopyToReg/CopyFromReg
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_S16_XACC_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s16_xacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// ESP.VMULAS.S16.XACC.ST.IP (_m version) - Store and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_S16_XACC_ST_IP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s16.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_S16_XACC_ST_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_s16_xacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

// ESP.VMULAS.S16.XACC.ST.XP (_m version) - Store and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_S16_XACC_ST_XP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.s16.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_S16_XACC_ST_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_s16_xacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

// ESP_VMULAS_S8_QACC: Multiply-accumulate to QACC with explicit passthru
// QACC_H:$qacc_h_in and QACC_L:$qacc_l_in are phantom operands (not shown in assembly string)
// for data flow tracking. They are explicit operands but hidden in assembly output.
let Defs = []<Register> in
def ESP_VMULAS_S8_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                 (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy),
                   "esp.vmulas.s8.qacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<16> qacc_l_l;      // QACC_L_LOW register (128-bit, v16i8)
  bits<16> qacc_l_h;       // QACC_L_HIGH register (128-bit, v16i8)
  bits<16> qacc_h_l;       // QACC_H_LOW register (128-bit, v16i8)
  bits<16> qacc_h_h;       // QACC_H_HIGH register (128-bit, v16i8)
  bits<16> qacc_l_l_in;    // QACC_L_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_l_h_in;    // QACC_L_HIGH passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_l_in;    // QACC_H_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_h_in;    // QACC_H_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMULAS_S8_QACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_s8_qacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_s8_qacc timm:$qx, timm:$qy)]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_S8_QACC_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s8.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L and QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_S8_QACC_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s8_qacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_S8_QACC_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                       (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s8.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L and QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_S8_QACC_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s8_qacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_S8_QACC_ST_IP: EspVInst<(outs GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qu, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s8.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_S8_QACC_ST_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_s8_qacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_S8_QACC_ST_XP: EspVInst<(outs GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qu, QR:$qx, QR:$qy, GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.vmulas.s8.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_S8_QACC_ST_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_s8_qacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

// ESP.VMULAS.S8.XACC (_m version) - Multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// ESP.VMULAS.S8.XACC (_m version) - Multiply-accumulate with explicit state passing
// Mixed model: XACC as struct {i32 (XACC[31:0]), i8 (XACC[39:32])}
// Note: Uses manual instruction selection (RISCVISelDAGToDAG.cpp) because Pattern
// cannot handle multiple outputs directly. Pattern matching is done in RISCVInstrInfoESP32P4Patterns.td
def ESP_VMULAS_S8_XACC: EspVInst<(outs XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), 
                                   (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy),
                   "esp.vmulas.s8.xacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: XACC input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_S8_XACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_s8_xacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_s8_xacc timm:$qx, timm:$qy)]>;

// ESP.VMULAS.S8.XACC.LD.IP (_m version) - Load and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_S8_XACC_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s8.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   // Lowering code handles XACC via CopyToReg/CopyFromReg
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_S8_XACC_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s8_xacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

// ESP.VMULAS.S8.XACC.LD.XP (_m version) - Load and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_S8_XACC_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s8.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   // Lowering code handles XACC via CopyToReg/CopyFromReg
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_S8_XACC_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s8_xacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// ESP.VMULAS.S8.XACC.ST.IP (_m version) - Store and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_S8_XACC_ST_IP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.s8.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_S8_XACC_ST_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_s8_xacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

// ESP.VMULAS.S8.XACC.ST.XP (_m version) - Store and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_S8_XACC_ST_XP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.s8.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_S8_XACC_ST_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_s8_xacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

// ESP_VMULAS_U16_QACC: Multiply-accumulate to QACC with explicit passthru
// QACC_H:$qacc_h_in and QACC_L:$qacc_l_in are phantom operands (not shown in assembly string)
// for data flow tracking. They are explicit operands but hidden in assembly output.
let Defs = []<Register> in
def ESP_VMULAS_U16_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                  (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy),
                   "esp.vmulas.u16.qacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<16> qacc_l_l;      // QACC_L_LOW register (128-bit, v16i8)
  bits<16> qacc_l_h;       // QACC_L_HIGH register (128-bit, v16i8)
  bits<16> qacc_h_l;       // QACC_H_LOW register (128-bit, v16i8)
  bits<16> qacc_h_h;       // QACC_H_HIGH register (128-bit, v16i8)
  bits<16> qacc_l_l_in;    // QACC_L_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_l_h_in;    // QACC_L_HIGH passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_l_in;    // QACC_H_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_h_in;    // QACC_H_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMULAS_U16_QACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_u16_qacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_u16_qacc timm:$qx, timm:$qy)]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_U16_QACC_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                         (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u16.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L and QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_U16_QACC_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u16_qacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_U16_QACC_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u16.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L and QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_U16_QACC_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u16_qacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_U16_QACC_ST_IP: EspVInst<(outs GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                         (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qu, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u16.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_U16_QACC_ST_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_u16_qacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_U16_QACC_ST_XP: EspVInst<(outs GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                         (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qu, QR:$qx, QR:$qy, GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.vmulas.u16.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_U16_QACC_ST_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_u16_qacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

// ESP.VMULAS.U16.XACC (_m version) - Multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// ESP.VMULAS.U16.XACC (_m version) - Multiply-accumulate with explicit state passing
// Mixed model: XACC as struct {i32 (XACC[31:0]), i8 (XACC[39:32])}
// Note: Uses manual instruction selection (RISCVISelDAGToDAG.cpp) because Pattern
// cannot handle multiple outputs directly. Pattern matching is done in RISCVInstrInfoESP32P4Patterns.td
def ESP_VMULAS_U16_XACC: EspVInst<(outs XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), 
                                   (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy),
                   "esp.vmulas.u16.xacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: XACC input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_U16_XACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_u16_xacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_u16_xacc timm:$qx, timm:$qy)]>;

// ESP.VMULAS.U16.XACC.LD.IP (_m version) - Load and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_U16_XACC_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u16.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   // Lowering code handles XACC via CopyToReg/CopyFromReg
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_U16_XACC_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u16_xacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

// ESP.VMULAS.U16.XACC.LD.XP (_m version) - Load and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_U16_XACC_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u16.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   // Lowering code handles XACC via CopyToReg/CopyFromReg
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_U16_XACC_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u16_xacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// ESP.VMULAS.U16.XACC.ST.IP (_m version) - Store and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_U16_XACC_ST_IP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u16.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_U16_XACC_ST_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_u16_xacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

// ESP.VMULAS.U16.XACC.ST.XP (_m version) - Store and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_U16_XACC_ST_XP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.u16.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_U16_XACC_ST_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_u16_xacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

// ESP_VMULAS_U8_QACC: Multiply-accumulate to QACC with explicit passthru
// QACC_H:$qacc_h_in and QACC_L:$qacc_l_in are phantom operands (not shown in assembly string)
// for data flow tracking. They are explicit operands but hidden in assembly output.
let Defs = []<Register> in
def ESP_VMULAS_U8_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                 (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy),
                   "esp.vmulas.u8.qacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<16> qacc_l_l;      // QACC_L_LOW register (128-bit, v16i8)
  bits<16> qacc_l_h;       // QACC_L_HIGH register (128-bit, v16i8)
  bits<16> qacc_h_l;       // QACC_H_LOW register (128-bit, v16i8)
  bits<16> qacc_h_h;       // QACC_H_HIGH register (128-bit, v16i8)
  bits<16> qacc_l_l_in;    // QACC_L_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_l_h_in;    // QACC_L_HIGH passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_l_in;    // QACC_H_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_h_in;    // QACC_H_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMULAS_U8_QACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_u8_qacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_u8_qacc timm:$qx, timm:$qy)]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_U8_QACC_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u8.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L and QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_U8_QACC_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u8_qacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_U8_QACC_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                       (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u8.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L and QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_U8_QACC_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u8_qacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_U8_QACC_ST_IP: EspVInst<(outs GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qu, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u8.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_U8_QACC_ST_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_u8_qacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_U8_QACC_ST_XP: EspVInst<(outs GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qu, QR:$qx, QR:$qy, GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.vmulas.u8.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_U8_QACC_ST_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_u8_qacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

// ESP.VMULAS.U8.XACC (_m version) - Multiply-accumulate with explicit state passing
// Mixed model: XACC as struct {i32 (XACC[31:0]), i8 (XACC[39:32])}
// ESP.VMULAS.U8.XACC (_m version) - Multiply-accumulate with explicit state passing
// Mixed model: XACC as struct {i32 (XACC[31:0]), i8 (XACC[39:32])}
// Note: Uses manual instruction selection (RISCVISelDAGToDAG.cpp) because Pattern
// cannot handle multiple outputs directly. Pattern matching is done in RISCVInstrInfoESP32P4Patterns.td
def ESP_VMULAS_U8_XACC: EspVInst<(outs XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), 
                                   (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy),
                   "esp.vmulas.u8.xacc\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: XACC input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_U8_XACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vmulas_u8_xacc_p $qx, $qy",
			[(int_riscv_esp_vmulas_u8_xacc timm:$qx, timm:$qy)]>;

// ESP.VMULAS.U8.XACC.LD.IP (_m version) - Load and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_U8_XACC_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u8.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   // Lowering code handles XACC via CopyToReg/CopyFromReg
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_U8_XACC_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u8_xacc_ld_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_ld_ip timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$off1616, timm:$qu))]>;

// ESP.VMULAS.U8.XACC.LD.XP (_m version) - Load and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_U8_XACC_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u8.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   // Lowering code handles XACC via CopyToReg/CopyFromReg
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_U8_XACC_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u8_xacc_ld_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// ESP.VMULAS.U8.XACC.ST.IP (_m version) - Store and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_U8_XACC_ST_IP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vmulas.u8.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<4> off1616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_U8_XACC_ST_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, offset_16_16:$off1616),
				"!esp_vmulas_u8_xacc_st_ip_p $qu, $rs1, $off1616, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_st_ip timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$off1616))]>;

// ESP.VMULAS.U8.XACC.ST.XP (_m version) - Store and multiply-accumulate with explicit state passing
// XACC is explicit input and output operand for chaining
// Note: Lowering code handles XACC via CopyToReg/CopyFromReg, so instruction uses implicit XACC
// but intrinsic takes explicit XACC passthru and returns new XACC
// Mixed model: XACC as {i32 low, i32 high}
def ESP_VMULAS_U8_XACC_ST_XP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmulas.u8.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy", 
                   []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_VMULAS_U8_XACC_ST_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1),
				"!esp_vmulas_u8_xacc_st_xp_p $qu, $rs1, $rs2, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// No implicit QACC_H_REG/QACC_L_REG references needed - similar to ESP_MOV_S16_QACC and ESP_VLD_128_IP
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_S16_QACC_LDBC_INCP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s16.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_S16_QACC_LDBC_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s16_qacc_ldbc_incp_p $qu, $rs1, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_ldbc_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_S8_QACC_LDBC_INCP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.s8.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_S8_QACC_LDBC_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_s8_qacc_ldbc_incp_p $qu, $rs1, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_ldbc_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_U16_QACC_LDBC_INCP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u16.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VMULAS_U16_QACC_LDBC_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u16_qacc_ldbc_incp_p $qu, $rs1, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_ldbc_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// QACC is passed as explicit 4x128-bit subregisters (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
let Defs = []<Register>, Uses = []<Register> in
def ESP_VMULAS_U8_QACC_LDBC_INCP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmulas.u8.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMULAS_U8_QACC_LDBC_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qu),
				"!esp_vmulas_u8_qacc_ldbc_incp_p $qu, $rs1, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_ldbc_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qu))]>;

// ESP_VSMULAS_S16_QACC: Multiply-accumulate to QACC with explicit passthru
// QACC_H:$qacc_h_in and QACC_L:$qacc_l_in are phantom operands (not shown in assembly string)
// for data flow tracking. They are explicit operands but hidden in assembly output.
let Defs = []<Register> in
def ESP_VSMULAS_S16_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                  (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, select_16:$sel16),
                   "esp.vsmulas.s16.qacc\t $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;
  bits<16> qacc_l_l;      // QACC_L_LOW register (128-bit, v16i8)
  bits<16> qacc_l_h;       // QACC_L_HIGH register (128-bit, v16i8)
  bits<16> qacc_h_l;       // QACC_H_LOW register (128-bit, v16i8)
  bits<16> qacc_h_h;       // QACC_H_HIGH register (128-bit, v16i8)
  bits<16> qacc_l_l_in;    // QACC_L_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_l_h_in;    // QACC_L_HIGH passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_l_in;    // QACC_H_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_h_in;    // QACC_H_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSMULAS_S16_QACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, select_16:$sel16),
				"!esp_vsmulas_s16_qacc_p $qx, $qy, $sel16",
			[(int_riscv_esp_vsmulas_s16_qacc timm:$qx, timm:$qy, timm:$sel16)]>;

let Defs = []<Register> in
def ESP_VSMULAS_S16_QACC_LD_INCP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_16:$sel16),
                   "esp.vsmulas.s16.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VSMULAS_S16_QACC_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_s16_qacc_ld_incp_p $qu, $rs1, $qx, $qy, $sel16",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_s16_qacc_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel16, timm:$qu))]>;

// ESP_VSMULAS_S8_QACC: Multiply-accumulate to QACC with explicit passthru
let Defs = []<Register> in
def ESP_VSMULAS_S8_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                  (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, select_16:$sel16),
                   "esp.vsmulas.s8.qacc\t $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;
  bits<16> qacc_l_l;      // QACC_L_LOW register (128-bit, v16i8)
  bits<16> qacc_l_h;       // QACC_L_HIGH register (128-bit, v16i8)
  bits<16> qacc_h_l;       // QACC_H_LOW register (128-bit, v16i8)
  bits<16> qacc_h_h;       // QACC_H_HIGH register (128-bit, v16i8)
  bits<16> qacc_l_l_in;    // QACC_L_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_l_h_in;    // QACC_L_HIGH passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_l_in;    // QACC_H_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_h_in;    // QACC_H_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSMULAS_S8_QACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, select_16:$sel16),
				"!esp_vsmulas_s8_qacc_p $qx, $qy, $sel16",
			[(int_riscv_esp_vsmulas_s8_qacc timm:$qx, timm:$qy, timm:$sel16)]>;

let Defs = []<Register> in
def ESP_VSMULAS_S8_QACC_LD_INCP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_16:$sel16),
                   "esp.vsmulas.s8.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VSMULAS_S8_QACC_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_s8_qacc_ld_incp_p $qu, $rs1, $qx, $qy, $sel16",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_s8_qacc_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel16, timm:$qu))]>;

// ESP_VSMULAS_U16_QACC: Multiply-accumulate to QACC with explicit passthru
let Defs = []<Register> in
def ESP_VSMULAS_U16_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                   (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, select_16:$sel16),
                   "esp.vsmulas.u16.qacc\t $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;
  bits<16> qacc_l_l;      // QACC_L_LOW register (128-bit, v16i8)
  bits<16> qacc_l_h;       // QACC_L_HIGH register (128-bit, v16i8)
  bits<16> qacc_h_l;       // QACC_H_LOW register (128-bit, v16i8)
  bits<16> qacc_h_h;       // QACC_H_HIGH register (128-bit, v16i8)
  bits<16> qacc_l_l_in;    // QACC_L_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_l_h_in;    // QACC_L_HIGH passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_l_in;    // QACC_H_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_h_in;    // QACC_H_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSMULAS_U16_QACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, select_16:$sel16),
				"!esp_vsmulas_u16_qacc_p $qx, $qy, $sel16",
			[(int_riscv_esp_vsmulas_u16_qacc timm:$qx, timm:$qy, timm:$sel16)]>;

let Defs = []<Register> in
def ESP_VSMULAS_U16_QACC_LD_INCP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_16:$sel16),
                   "esp.vsmulas.u16.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // QACC_L and QACC_H tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $qacc_l_l = $qacc_l_l_in, $qacc_l_h = $qacc_l_h_in, $qacc_h_l = $qacc_h_l_in, $qacc_h_h = $qacc_h_h_in";
  
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
def ESP_VSMULAS_U16_QACC_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_u16_qacc_ld_incp_p $qu, $rs1, $qx, $qy, $sel16",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_u16_qacc_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel16, timm:$qu))]>;

// ESP_VSMULAS_U8_QACC: Multiply-accumulate to QACC with explicit passthru
let Defs = []<Register> in
def ESP_VSMULAS_U8_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                  (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, select_16:$sel16),
                   "esp.vsmulas.u8.qacc\t $qx, $qy, $sel16", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;
  bits<16> qacc_l_l;      // QACC_L_LOW register (128-bit, v16i8)
  bits<16> qacc_l_h;       // QACC_L_HIGH register (128-bit, v16i8)
  bits<16> qacc_h_l;       // QACC_H_LOW register (128-bit, v16i8)
  bits<16> qacc_h_h;       // QACC_H_HIGH register (128-bit, v16i8)
  bits<16> qacc_l_l_in;    // QACC_L_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_l_h_in;    // QACC_L_HIGH passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_l_in;    // QACC_H_LOW passthru (phantom operand, not shown in assembly)
  bits<16> qacc_h_h_in;    // QACC_H_HIGH passthru (phantom operand, not shown in assembly)
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSMULAS_U8_QACC_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, select_16:$sel16),
				"!esp_vsmulas_u8_qacc_p $qx, $qy, $sel16",
			[(int_riscv_esp_vsmulas_u8_qacc timm:$qx, timm:$qy, timm:$sel16)]>;

let Defs = []<Register> in
def ESP_VSMULAS_U8_QACC_LD_INCP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), 
                                        (ins QACC_L_LOW:$qacc_l_l_in, QACC_L_HIGH:$qacc_l_h_in, QACC_H_LOW:$qacc_h_l_in, QACC_H_HIGH:$qacc_h_h_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_16:$sel16),
                   "esp.vsmulas.u8.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VSMULAS_U8_QACC_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_u8_qacc_ld_incp_p $qu, $rs1, $qx, $qy, $sel16",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_u8_qacc_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel16, timm:$qu))]>;

// ESP.CMUL.S16 with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_S16: EspVInst<(outs QR:$qz), (ins QR:$qz_in, QR:$qx, QR:$qy, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.s16\t $qz, $qx, $qy, $sel4",
                   // SAR is explicit in intrinsic and instruction (explicit state passing)
                   [(set QR:$qz, (int_riscv_esp_cmul_s16_m QR:$qz_in, QR:$qx, QR:$qy, timm:$sel4, SARReg:$sar_in))]>
{
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  let Constraints = "$qz = $qz_in";
  
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
def ESP_CMUL_S16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, select_4:$sel4, imm8:$qz),
				"!esp_cmul_s16_p $qz, $qx, $qy, $sel4",
			[(int_riscv_esp_cmul_s16 timm:$qx, timm:$qy, timm:$sel4, timm:$qz)]>;

// ESP_CMUL_S16_LD_INCP now matches SDNode directly, no pseudo instruction needed

// ESP.CMUL.S16.LD.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_S16_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qz_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.s16.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", 
                   [(set QR:$qz, QR:$qu, GPRPIE:$rs1r, (riscv_esp_cmul_s16_ld_incp_m QR:$qz_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
  let Constraints = !interleave(["$rs1r = $rs1", "$qz = $qz_in"], ",");
  
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
def ESP_CMUL_S16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_4:$sel4, imm8:$qz, imm8:$qu),
				"!esp_cmul_s16_ld_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel4, timm:$qz, timm:$qu))]>;

// ESP.CMUL.S16.ST.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_S16_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qz_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.s16.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", 
                   [(set QR:$qz, GPRPIE:$rs1r, (riscv_esp_cmul_s16_st_incp_m QR:$qz_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
  let Constraints = !interleave(["$rs1r = $rs1", "$qz = $qz_in"], ",");
  
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
def ESP_CMUL_S16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, select_4:$sel4, imm8:$qz),
				"!esp_cmul_s16_st_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sel4, timm:$qz))]>;

// ESP.CMUL.S8 with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_S8: EspVInst<(outs QR:$qz), (ins QR:$qz_in, QR:$qx, QR:$qy, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.s8\t $qz, $qx, $qy, $sel4",
                   // SAR is explicit in intrinsic and instruction (explicit state passing)
                   [(set QR:$qz, (int_riscv_esp_cmul_s8_m QR:$qz_in, QR:$qx, QR:$qy, timm:$sel4, SARReg:$sar_in))]>
{
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  let Constraints = "$qz = $qz_in";
  
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
def ESP_CMUL_S8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, select_4:$sel4, imm8:$qz),
				"!esp_cmul_s8_p $qz, $qx, $qy, $sel4",
			[(int_riscv_esp_cmul_s8 timm:$qx, timm:$qy, timm:$sel4, timm:$qz)]>;

// ESP.CMUL.S8.LD.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_S8_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qz_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.s8.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", 
                   [(set QR:$qz, QR:$qu, GPRPIE:$rs1r, (riscv_esp_cmul_s8_ld_incp_m QR:$qz_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
  let Constraints = !interleave(["$rs1r = $rs1", "$qz = $qz_in"], ",");
  
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
def ESP_CMUL_S8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_4:$sel4, imm8:$qz, imm8:$qu),
				"!esp_cmul_s8_ld_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel4, timm:$qz, timm:$qu))]>;

// ESP.CMUL.S8.ST.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_S8_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qz_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.s8.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", 
                   [(set QR:$qz, GPRPIE:$rs1r, (riscv_esp_cmul_s8_st_incp_m QR:$qz_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
  let Constraints = !interleave(["$rs1r = $rs1", "$qz = $qz_in"], ",");
  
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
def ESP_CMUL_S8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, select_4:$sel4, imm8:$qz),
				"!esp_cmul_s8_st_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sel4, timm:$qz))]>;

// ESP.CMUL.U16 with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_U16: EspVInst<(outs QR:$qz), (ins QR:$qz_in, QR:$qx, QR:$qy, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.u16\t $qz, $qx, $qy, $sel4",
                   // SAR is explicit in intrinsic and instruction (explicit state passing)
                   [(set QR:$qz, (int_riscv_esp_cmul_u16_m QR:$qz_in, QR:$qx, QR:$qy, timm:$sel4, SARReg:$sar_in))]>
{
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  let Constraints = "$qz = $qz_in";
  
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
def ESP_CMUL_U16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, select_4:$sel4, imm8:$qz),
				"!esp_cmul_u16_p $qz, $qx, $qy, $sel4",
			[(int_riscv_esp_cmul_u16 timm:$qx, timm:$qy, timm:$sel4, timm:$qz)]>;

// ESP.CMUL.U16.LD.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_U16_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qz_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.u16.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", 
                   [(set QR:$qz, QR:$qu, GPRPIE:$rs1r, (riscv_esp_cmul_u16_ld_incp_m QR:$qz_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
  let Constraints = !interleave(["$rs1r = $rs1", "$qz = $qz_in"], ",");
  
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
def ESP_CMUL_U16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_4:$sel4, imm8:$qz, imm8:$qu),
				"!esp_cmul_u16_ld_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel4, timm:$qz, timm:$qu))]>;

// ESP.CMUL.U16.ST.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_U16_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qz_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.u16.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", 
                   [(set QR:$qz, GPRPIE:$rs1r, (riscv_esp_cmul_u16_st_incp_m QR:$qz_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
  let Constraints = !interleave(["$rs1r = $rs1", "$qz = $qz_in"], ",");
  
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
def ESP_CMUL_U16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, select_4:$sel4, imm8:$qz),
				"!esp_cmul_u16_st_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sel4, timm:$qz))]>;

// ESP.CMUL.U8 with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_U8: EspVInst<(outs QR:$qz), (ins QR:$qz_in, QR:$qx, QR:$qy, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.u8\t $qz, $qx, $qy, $sel4",
                   // SAR is explicit in intrinsic and instruction (explicit state passing)
                   [(set QR:$qz, (int_riscv_esp_cmul_u8_m QR:$qz_in, QR:$qx, QR:$qy, timm:$sel4, SARReg:$sar_in))]>
{
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  let Constraints = "$qz = $qz_in";
  
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
def ESP_CMUL_U8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, select_4:$sel4, imm8:$qz),
				"!esp_cmul_u8_p $qz, $qx, $qy, $sel4",
			[(int_riscv_esp_cmul_u8 timm:$qx, timm:$qy, timm:$sel4, timm:$qz)]>;

// ESP.CMUL.U8.LD.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_U8_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qz_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.u8.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", 
                   [(set QR:$qz, QR:$qu, GPRPIE:$rs1r, (riscv_esp_cmul_u8_ld_incp_m QR:$qz_in, QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
  let Constraints = !interleave(["$rs1r = $rs1", "$qz = $qz_in"], ",");
  
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
def ESP_CMUL_U8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_4:$sel4, imm8:$qz, imm8:$qu),
				"!esp_cmul_u8_ld_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel4, timm:$qz, timm:$qu))]>;

// ESP.CMUL.U8.ST.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_CMUL_U8_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qz_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in),
                   "esp.cmul.u8.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4", 
                   [(set QR:$qz, GPRPIE:$rs1r, (riscv_esp_cmul_u8_st_incp_m QR:$qz_in, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
  let Constraints = !interleave(["$rs1r = $rs1", "$qz = $qz_in"], ",");
  
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
def ESP_CMUL_U8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, select_4:$sel4, imm8:$qz),
				"!esp_cmul_u8_st_incp_p $qu, $rs1, $qz, $qx, $qy, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sel4, timm:$qz))]>;

def ESP_MAX_S16_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.s16.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MAX_S16_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_s16_a_p $qw, $rd",
			[(int_riscv_esp_max_s16_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_S32_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.s32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MAX_S32_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_s32_a_p $qw, $rd",
			[(int_riscv_esp_max_s32_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_S8_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.s8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MAX_S8_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_s8_a_p $qw, $rd",
			[(int_riscv_esp_max_s8_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_U16_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.u16.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MAX_U16_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_u16_a_p $qw, $rd",
			[(int_riscv_esp_max_u16_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_U32_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.u32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MAX_U32_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_u32_a_p $qw, $rd",
			[(int_riscv_esp_max_u32_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_U8_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.u8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MAX_U8_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_u8_a_p $qw, $rd",
			[(int_riscv_esp_max_u8_a timm:$qw, GPRPIE:$rd)]>;


def ESP_MIN_S16_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.s16.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MIN_S16_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_s16_a_p $qw, $rd",
			[(int_riscv_esp_min_s16_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_S32_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.s32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MIN_S32_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_s32_a_p $qw, $rd",
			[(int_riscv_esp_min_s32_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_S8_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.s8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MIN_S8_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_s8_a_p $qw, $rd",
			[(int_riscv_esp_min_s8_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_U16_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.u16.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MIN_U16_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_u16_a_p $qw, $rd",
			[(int_riscv_esp_min_u16_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_U32_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.u32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MIN_U32_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_u32_a_p $qw, $rd",
			[(int_riscv_esp_min_u32_a timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_U8_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.u8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MIN_U8_A_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_u8_a_p $qw, $rd",
			[(int_riscv_esp_min_u8_a timm:$qw, GPRPIE:$rd)]>;

// Support vector reduce max patterns - handling vecreduce_smax/umax to ESP32P4 instructions
class EspVectorReduceSmaxDirectPat<ValueType vt, ValueType result_vt, EspVInst Inst>
    : Pat<(result_vt (vecreduce_smax (vt QR:$qw))),
          (Inst QR:$qw)>;

class EspVectorReduceUmaxDirectPat<ValueType vt, ValueType result_vt, EspVInst Inst>
    : Pat<(result_vt (vecreduce_umax (vt QR:$qw))),
          (Inst QR:$qw)>;

class EspVectorReduceSminDirectPat<ValueType vt, ValueType result_vt, EspVInst Inst>
    : Pat<(result_vt (vecreduce_smin (vt QR:$qw))),
          (Inst QR:$qw)>;

class EspVectorReduceUminDirectPat<ValueType vt, ValueType result_vt, EspVInst Inst>
    : Pat<(result_vt (vecreduce_umin (vt QR:$qw))),
          (Inst QR:$qw)>;

// Pattern match vecreduce_smax to ESP_MAX instructions
def : EspVectorReduceSmaxDirectPat<v16i8, i32, ESP_MAX_S8_A>;
def : EspVectorReduceSmaxDirectPat<v8i16, i32, ESP_MAX_S16_A>;
def : EspVectorReduceSmaxDirectPat<v4i32, i32, ESP_MAX_S32_A>;

// Pattern match vecreduce_umax to ESP_MAX instructions
def : EspVectorReduceUmaxDirectPat<v16i8, i32, ESP_MAX_U8_A>;
def : EspVectorReduceUmaxDirectPat<v8i16, i32, ESP_MAX_U16_A>;
def : EspVectorReduceUmaxDirectPat<v4i32, i32, ESP_MAX_U32_A>;

// Pattern match vecreduce_smin to ESP_MIN instructions
def : EspVectorReduceSminDirectPat<v16i8, i32, ESP_MIN_S8_A>;
def : EspVectorReduceSminDirectPat<v8i16, i32, ESP_MIN_S16_A>;
def : EspVectorReduceSminDirectPat<v4i32, i32, ESP_MIN_S32_A>;

// Pattern match vecreduce_umin to ESP_MIN instructions
def : EspVectorReduceUminDirectPat<v16i8, i32, ESP_MIN_U8_A>;
def : EspVectorReduceUminDirectPat<v8i16, i32, ESP_MIN_U16_A>;
def : EspVectorReduceUminDirectPat<v4i32, i32, ESP_MIN_U32_A>;

def ESP_VABS_16: EspVInst<(outs QR:$qv), (ins QR:$qy),
                   "esp.vabs.16\t $qv, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vabs_16_m QR:$qy))]> 
{
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VABS_16_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qv),
				"!esp_vabs_16_p $qv, $qy",
			[(int_riscv_esp_vabs_16 timm:$qy, timm:$qv)]>;

def ESP_VABS_32: EspVInst<(outs QR:$qv), (ins QR:$qy),
                   "esp.vabs.32\t $qv, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vabs_32_m QR:$qy))]> 
{
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VABS_32_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qv),
				"!esp_vabs_32_p $qv, $qy",
			[(int_riscv_esp_vabs_32 timm:$qy, timm:$qv)]>;

def ESP_VABS_8: EspVInst<(outs QR:$qv), (ins QR:$qy),
                   "esp.vabs.8\t $qv, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vabs_8_m QR:$qy))]> 
{
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VABS_8_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qv),
				"!esp_vabs_8_p $qv, $qy",
			[(int_riscv_esp_vabs_8 timm:$qy, timm:$qv)]>;

def ESP_VADD_S16: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.s16\t $qv, $qx, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vadd_s16_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_S16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_s16_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_s16 timm:$qx, timm:$qy, timm:$qv)]>;

// VADD pseudo instructions (_m versions)
// These match intrinsics directly to instructions
// Note: Direct match to SDNode, no custom inserter needed
def ESP_VADD_S16_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.s16.ld.incp\t $qu, $rs1, $qv, $qx, $qy",
                   [(set QR:$qv, QR:$qu, GPRPIE:$rs1r,
                      (riscv_esp_vadd_s16_ld_incp_m QR:$qx, QR:$qy, GPRPIE:$rs1))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VADD_S16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_s16_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_S16_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.s16.st.incp\t $qu, $rs1, $qv, $qx, $qy",
                   [(set QR:$qv, GPRPIE:$rs1r,
                      (riscv_esp_vadd_s16_st_incp_m QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_S16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_s16_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;



def ESP_VADD_S32: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.s32\t $qv, $qx, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vadd_s32_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_S32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_s32_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_s32 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VADD_S32_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.s32.ld.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VADD_S32_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_s32_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_S32_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.s32.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_S32_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_s32_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;



def ESP_VADD_S8: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.s8\t $qv, $qx, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vadd_s8_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_S8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_s8_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_s8 timm:$qx, timm:$qy, timm:$qv)]>;


def ESP_VADD_S8_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.s8.ld.incp\t $qu, $rs1, $qv, $qx, $qy", 
                   [(set QR:$qv, QR:$qu, GPRPIE:$rs1r, 
                      (riscv_esp_vadd_s8_ld_incp_m QR:$qx, QR:$qy, GPRPIE:$rs1))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VADD_S8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_s8_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_S8_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.s8.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   [(set QR:$qv, GPRPIE:$rs1r,
                      (riscv_esp_vadd_s8_st_incp_m QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_S8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_s8_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;



def ESP_VADD_U16: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.u16\t $qv, $qx, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vadd_u16_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_U16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_u16_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_u16 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VADD_U16_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.u16.ld.incp	 $qu, $rs1, $qv, $qx, $qy",
                   [(set QR:$qv, QR:$qu, GPRPIE:$rs1r,
                      (riscv_esp_vadd_u16_ld_incp_m QR:$qx, QR:$qy, GPRPIE:$rs1))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VADD_U16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_u16_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_U16_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.u16.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   [(set QR:$qv, GPRPIE:$rs1r,
                      (riscv_esp_vadd_u16_st_incp_m QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_U16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_u16_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;



def ESP_VADD_U32: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.u32\t $qv, $qx, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vadd_u32_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_U32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_u32_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_u32 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VADD_U32_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.u32.ld.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VADD_U32_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_u32_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_U32_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.u32.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_U32_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_u32_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;



def ESP_VADD_U8: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vadd.u8\t $qv, $qx, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vadd_u8_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_U8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vadd_u8_p $qv, $qx, $qy",
			[(int_riscv_esp_vadd_u8 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VADD_U8_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vadd.u8.ld.incp	 $qu, $rs1, $qv, $qx, $qy",
                   [(set QR:$qv, QR:$qu, GPRPIE:$rs1r,
                      (riscv_esp_vadd_u8_ld_incp_m QR:$qx, QR:$qy, GPRPIE:$rs1))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VADD_U8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vadd_u8_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VADD_U8_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vadd.u8.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   [(set QR:$qv, GPRPIE:$rs1r,
                      (riscv_esp_vadd_u8_st_incp_m QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VADD_U8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vadd_u8_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;



def ESP_VCLAMP_S16: EspVInst<(outs QR:$qz), (ins QR:$qx, select_16:$sel16),
                   "esp.vclamp.s16\t $qz, $qx, $sel16", 
                   [(set QR:$qz, (int_riscv_esp_vclamp_s16_m QR:$qx, timm:$sel16))]> 
{
  bits<3> qx;
  bits<4> sel16;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCLAMP_S16_P : PseudoEsp<(outs), (ins imm8:$qx, select_16:$sel16, imm8:$qz),
				"!esp_vclamp_s16_p $qz, $qx, $sel16",
			[(int_riscv_esp_vclamp_s16 timm:$qx, timm:$sel16, timm:$qz)]>;

// Note: Scalar instructions (ADDX2, ADDX4, SUBX2, SUBX4, SAT) are already
// handled through ClangBuiltin in IntrinsicsRISCVESP32P4.td, so they don't
// need _m version pseudo instruction patterns here.

def ESP_VMAX_S16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.s16\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vmax_s16_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_S16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_S16_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.s16.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_S16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_s16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_S16_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.s16.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_S16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_s16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


def ESP_VMAX_S32: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.s32\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vmax_s32_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_S32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_s32_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_s32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_S32_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.s32.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_S32_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_s32_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_S32_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.s32.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_S32_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_s32_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


def ESP_VMAX_S8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.s8\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vmax_s8_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_S8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_S8_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.s8.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_S8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_s8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_S8_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.s8.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_S8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_s8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


def ESP_VMAX_U16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.u16\t $qz, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_U16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_U16_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.u16.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_U16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_u16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_U16_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.u16.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_U16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_u16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


def ESP_VMAX_U32: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.u32\t $qz, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_U32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_u32_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_u32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_U32_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.u32.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_U32_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_u32_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_U32_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.u32.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_U32_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_u32_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


def ESP_VMAX_U8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmax.u8\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vmax_u8_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_U8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_u8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_U8_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmax.u8.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_U8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_u8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_U8_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmax.u8.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMAX_U8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_u8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


def ESP_VMIN_S16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.s16\t $qz, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_S16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_S16_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.s16.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_S16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_s16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_S16_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.s16.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_S16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_s16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


def ESP_VMIN_S32: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.s32\t $qz, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_S32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_s32_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_s32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_S32_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.s32.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_S32_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_s32_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_S32_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.s32.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_S32_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_s32_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


def ESP_VMIN_S8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.s8\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vmin_s8_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_S8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_S8_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.s8.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_S8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_s8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_S8_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.s8.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_S8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_s8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


def ESP_VMIN_U16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.u16\t $qz, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_U16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_U16_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.u16.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_U16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_u16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_U16_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.u16.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_U16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_u16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;



def ESP_VMIN_U32: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.u32\t $qz, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_U32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_u32_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_u32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_U32_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.u32.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_U32_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_u32_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_U32_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.u32.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_U32_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_u32_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;



def ESP_VMIN_U8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vmin.u8\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vmin_u8_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_U8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_u8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_U8_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vmin.u8.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_U8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_u8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_U8_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vmin.u8.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMIN_U8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_u8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


// ESP.VMUL.S16 reads from SAR register (SAR is input source)
// For explicit state passing, SAR (64-bit) is passed as explicit parameter in intrinsic
// Instruction: qz = (qx * qy) >> SAR[5:0] (hardware extracts only low 6 bits from SAR)
def ESP_VMUL_S16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy, SARReg:$sar_in),
                   "esp.vmul.s16\t $qz, $qx, $qy", 
                   // SAR is explicit in intrinsic (SAR is input parameter, i32)
                   // Hardware automatically extracts SAR[5:0] for shift amount
                   // Pattern: use i32 instead of SARReg to match SDNode operand type
                   // LLVM will insert COPY instruction to convert GPR to SARReg
                   [(set QR:$qz, (int_riscv_esp_vmul_s16_m QR:$qx, QR:$qy, i32:$sar_in))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_S16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmul_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmul_s16 timm:$qx, timm:$qy, timm:$qz)]>;


// ESP.VMUL.S16.LD.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_VMUL_S16_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, SARReg:$sar_in),
                   "esp.vmul.s16.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_S16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmul_s16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;


// ESP.VMUL.S16.S8XS8 reads from SAR register (SAR is input source)
// For explicit state passing, SAR (64-bit) is passed as explicit parameter in intrinsic
// Instruction: qz, qv = (qx * qy) >> SAR[5:0] (hardware extracts only low 6 bits from SAR)
def ESP_VMUL_S16_S8XS8: EspVInst<(outs QR:$qz, QR:$qv), (ins QR:$qx, QR:$qy, SARReg:$sar_in),
                   "esp.vmul.s16.s8xs8\t $qz, $qv, $qx, $qy", 
                   // SAR is explicit in intrinsic (SAR is input parameter, 64-bit)
                   // Hardware automatically extracts SAR[5:0] for shift amount
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_S16_S8XS8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz, imm8:$qv),
				"!esp_vmul_s16_s8xs8_p $qz, $qv, $qx, $qy",
			[(int_riscv_esp_vmul_s16_s8xs8 timm:$qx, timm:$qy, timm:$qz, timm:$qv)]>;


// ESP.VMUL.S16.ST.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
// Note: qv is output (qz), not input, so it's not in the pattern match
def ESP_VMUL_S16_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, SARReg:$sar_in),
                   "esp.vmul.s16.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_S16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmul_s16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


// ESP.VMUL.S32.S16XS16 reads from SAR register (SAR is input source)
// For explicit state passing, SAR (64-bit) is passed as explicit parameter in intrinsic
// Instruction: qz, qv = (qx * qy) >> SAR[5:0] (hardware extracts only low 6 bits from SAR)
def ESP_VMUL_S32_S16XS16: EspVInst<(outs QR:$qz, QR:$qv), (ins QR:$qx, QR:$qy, SARReg:$sar_in),
                   "esp.vmul.s32.s16xs16\t $qz, $qv, $qx, $qy", 
                   // SAR is explicit in intrinsic (SAR is input parameter, 64-bit)
                   // Hardware automatically extracts SAR[5:0] for shift amount
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_S32_S16XS16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz, imm8:$qv),
				"!esp_vmul_s32_s16xs16_p $qz, $qv, $qx, $qy",
			[(int_riscv_esp_vmul_s32_s16xs16 timm:$qx, timm:$qy, timm:$qz, timm:$qv)]>;

// ESP.VMUL.S8 reads from SAR register (SAR is input source)
// For explicit state passing, SAR (64-bit) is passed as explicit parameter in intrinsic
// Instruction: qz = (qx * qy) >> SAR[5:0] (hardware extracts only low 6 bits from SAR)
def ESP_VMUL_S8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy, SARReg:$sar_in),
                   "esp.vmul.s8\t $qz, $qx, $qy", 
                   // SAR is explicit in intrinsic (SAR is input parameter, i32)
                   // Hardware automatically extracts SAR[5:0] for shift amount
                   [(set QR:$qz, (int_riscv_esp_vmul_s8_m QR:$qx, QR:$qy, SARReg:$sar_in))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_S8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmul_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmul_s8 timm:$qx, timm:$qy, timm:$qz)]>;


// ESP.VMUL.S8.LD.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_VMUL_S8_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, SARReg:$sar_in),
                   "esp.vmul.s8.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_S8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmul_s8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;


// ESP.VMUL.S8.ST.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_VMUL_S8_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, SARReg:$sar_in),
                   "esp.vmul.s8.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_S8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmul_s8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


// ESP.VMUL.U16 reads from SAR register (SAR is input source)
// For explicit state passing, SAR (64-bit) is passed as explicit parameter in intrinsic
// Instruction: qz = (qx * qy) >> SAR[5:0] (hardware extracts only low 6 bits from SAR)
def ESP_VMUL_U16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy, SARReg:$sar_in),
                   "esp.vmul.u16\t $qz, $qx, $qy", 
                   // SAR is explicit in intrinsic (SAR is input parameter, 64-bit)
                   // Hardware automatically extracts SAR[5:0] for shift amount
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_U16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmul_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vmul_u16 timm:$qx, timm:$qy, timm:$qz)]>;


// ESP.VMUL.U16.LD.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_VMUL_U16_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, SARReg:$sar_in),
                   "esp.vmul.u16.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_U16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmul_u16_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;


// ESP.VMUL.U16.ST.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_VMUL_U16_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, SARReg:$sar_in),
                   "esp.vmul.u16.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_U16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmul_u16_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


// ESP.VMUL.U8 reads from SAR register (SAR is input source)
// For explicit state passing, SAR (64-bit) is passed as explicit parameter in intrinsic
// Instruction: qz = (qx * qy) >> SAR[5:0] (hardware extracts only low 6 bits from SAR)
def ESP_VMUL_U8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy, SARReg:$sar_in),
                   "esp.vmul.u8\t $qz, $qx, $qy", 
                   // SAR is explicit in intrinsic (SAR is input parameter, i32)
                   // Hardware automatically extracts SAR[5:0] for shift amount
                   [(set QR:$qz, (int_riscv_esp_vmul_u8_m QR:$qx, QR:$qy, SARReg:$sar_in))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_U8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmul_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vmul_u8 timm:$qx, timm:$qy, timm:$qz)]>;


// ESP.VMUL.U8.LD.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_VMUL_U8_LD_INCP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, SARReg:$sar_in),
                   "esp.vmul.u8.ld.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qz;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_U8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmul_u8_ld_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;


// ESP.VMUL.U8.ST.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_VMUL_U8_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, SARReg:$sar_in),
                   "esp.vmul.u8.st.incp\t $qu, $rs1, $qz, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VMUL_U8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmul_u8_st_incp_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;


def ESP_VPRELU_S16: EspVInst<(outs QR:$qz), (ins GPRPIE:$rs1, QR:$qx, QR:$qy),
                   "esp.vprelu.s16\t $qz, $qy, $qx, $rs1", 
                   [(set QR:$qz, (int_riscv_esp_vprelu_s16_m QR:$qx, QR:$qy, GPRPIE:$rs1))]> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VPRELU_S16_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vprelu_s16_p $qz, $qy, $qx, $rs1",
			[(int_riscv_esp_vprelu_s16 GPRPIE:$rs1, timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VPRELU_S8: EspVInst<(outs QR:$qz), (ins GPRPIE:$rs1, QR:$qx, QR:$qy),
                   "esp.vprelu.s8\t $qz, $qy, $qx, $rs1", 
                   [(set QR:$qz, (int_riscv_esp_vprelu_s8_m QR:$qx, QR:$qy, GPRPIE:$rs1))]> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VPRELU_S8_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vprelu_s8_p $qz, $qy, $qx, $rs1",
			[(int_riscv_esp_vprelu_s8 GPRPIE:$rs1, timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VRELU_S16: EspVInst<(outs QR:$qyr), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy),
                   "esp.vrelu.s16\t $qy, $rs2, $rs1", 
                   []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qy;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VRELU_S16_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy),
				"!esp_vrelu_s16_p $qy, $rs2, $rs1",
			[(int_riscv_esp_vrelu_s16 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy)]>;

def ESP_VRELU_S8: EspVInst<(outs QR:$qyr), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy),
                   "esp.vrelu.s8\t $qy, $rs2, $rs1", 
                   []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qy;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VRELU_S8_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy),
				"!esp_vrelu_s8_p $qy, $rs2, $rs1",
			[(int_riscv_esp_vrelu_s8 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy)]>;

def ESP_VSADDS_S16: EspVInst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vsadds.s16\t $qv, $qx, $rs1", 
                   [(set QR:$qv, (int_riscv_esp_vsadds_s16_m QR:$qx, GPRPIE:$rs1))]> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSADDS_S16_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vsadds_s16_p $qv, $qx, $rs1",
			[(int_riscv_esp_vsadds_s16 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSADDS_S8: EspVInst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vsadds.s8\t $qv, $qx, $rs1", 
                   [(set QR:$qv, (int_riscv_esp_vsadds_s8_m QR:$qx, GPRPIE:$rs1))]> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSADDS_S8_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vsadds_s8_p $qv, $qx, $rs1",
			[(int_riscv_esp_vsadds_s8 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSADDS_U16: EspVInst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vsadds.u16\t $qv, $qx, $rs1", 
                   [(set QR:$qv, (int_riscv_esp_vsadds_u16_m QR:$qx, GPRPIE:$rs1))]> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSADDS_U16_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vsadds_u16_p $qv, $qx, $rs1",
			[(int_riscv_esp_vsadds_u16 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSADDS_U8: EspVInst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vsadds.u8\t $qv, $qx, $rs1", 
                   [(set QR:$qv, (int_riscv_esp_vsadds_u8_m QR:$qx, GPRPIE:$rs1))]> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSADDS_U8_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vsadds_u8_p $qv, $qx, $rs1",
			[(int_riscv_esp_vsadds_u8 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSAT_S16: EspVInst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.s16\t $qz, $qx, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSAT_S16_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_s16_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_s16 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_S32: EspVInst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.s32\t $qz, $qx, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSAT_S32_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_s32_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_s32 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_S8: EspVInst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.s8\t $qz, $qx, $rs1, $rs2", 
                   [(set QR:$qz, (int_riscv_esp_vsat_s8_m QR:$qx, GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSAT_S8_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_s8_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_s8 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_U16: EspVInst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.u16\t $qz, $qx, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSAT_U16_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_u16_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_u16 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_U32: EspVInst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.u32\t $qz, $qx, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSAT_U32_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_u32_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_u32 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_U8: EspVInst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
                   "esp.vsat.u8\t $qz, $qx, $rs1, $rs2", 
                   [(set QR:$qz, (int_riscv_esp_vsat_u8_m QR:$qx, GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSAT_U8_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_u8_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_u8 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSSUBS_S16: EspVInst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vssubs.s16\t $qv, $qx, $rs1", 
                   [(set QR:$qv, (int_riscv_esp_vssubs_s16_m QR:$qx, GPRPIE:$rs1))]> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSSUBS_S16_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vssubs_s16_p $qv, $qx, $rs1",
			[(int_riscv_esp_vssubs_s16 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSSUBS_S8: EspVInst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vssubs.s8\t $qv, $qx, $rs1", 
                   [(set QR:$qv, (int_riscv_esp_vssubs_s8_m QR:$qx, GPRPIE:$rs1))]> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSSUBS_S8_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vssubs_s8_p $qv, $qx, $rs1",
			[(int_riscv_esp_vssubs_s8 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSSUBS_U16: EspVInst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vssubs.u16\t $qv, $qx, $rs1", 
                   [(set QR:$qv, (int_riscv_esp_vssubs_u16_m QR:$qx, GPRPIE:$rs1))]> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSSUBS_U16_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vssubs_u16_p $qv, $qx, $rs1",
			[(int_riscv_esp_vssubs_u16 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSSUBS_U8: EspVInst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx),
                   "esp.vssubs.u8\t $qv, $qx, $rs1", 
                   [(set QR:$qv, (int_riscv_esp_vssubs_u8_m QR:$qx, GPRPIE:$rs1))]> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSSUBS_U8_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qv),
				"!esp_vssubs_u8_p $qv, $qx, $rs1",
			[(int_riscv_esp_vssubs_u8 GPRPIE:$rs1, timm:$qx, timm:$qv)]>;

def ESP_VSUB_S16: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.s16\t $qv, $qx, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vsub_s16_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_S16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_s16_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_s16 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_S16_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.s16.ld.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_S16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_s16_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_S16_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.s16.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_S16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_s16_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;


def ESP_VSUB_S32: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.s32\t $qv, $qx, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vsub_s32_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_S32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_s32_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_s32 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_S32_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.s32.ld.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_S32_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_s32_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_S32_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.s32.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_S32_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_s32_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;


def ESP_VSUB_S8: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.s8\t $qv, $qx, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vsub_s8_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_S8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_s8_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_s8 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_S8_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.s8.ld.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_S8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_s8_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_S8_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.s8.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_S8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_s8_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;


def ESP_VSUB_U16: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.u16\t $qv, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_U16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_u16_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_u16 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_U16_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.u16.ld.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_U16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_u16_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u16_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_U16_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.u16.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_U16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_u16_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u16_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;


def ESP_VSUB_U32: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.u32\t $qv, $qx, $qy", 
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_U32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_u32_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_u32 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_U32_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.u32.ld.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_U32_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_u32_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u32_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_U32_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.u32.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_U32_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_u32_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u32_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;


def ESP_VSUB_U8: EspVInst<(outs QR:$qv), (ins QR:$qx, QR:$qy),
                   "esp.vsub.u8\t $qv, $qx, $qy", 
                   [(set QR:$qv, (int_riscv_esp_vsub_u8_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_U8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv),
				"!esp_vsub_u8_p $qv, $qx, $qy",
			[(int_riscv_esp_vsub_u8 timm:$qx, timm:$qy, timm:$qv)]>;

def ESP_VSUB_U8_LD_INCP: EspVInst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
                   "esp.vsub.u8.ld.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<3> qv;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_U8_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qv, imm8:$qu),
				"!esp_vsub_u8_ld_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u8_ld_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qv, timm:$qu))]>;

def ESP_VSUB_U8_ST_INCP: EspVInst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
                   "esp.vsub.u8.st.incp	 $qu, $rs1, $qv, $qx, $qy",
                   []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<3> qv;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSUB_U8_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qv),
				"!esp_vsub_u8_st_incp_p $qu, $rs1, $qv, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u8_st_incp timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qv))]>;


def ESP_ADDX2: EspVInst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.addx2\t $rd, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_ADDX2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd),
				"!esp_addx2_p $rd, $rs1, $rs2",
			[(int_riscv_esp_addx2 GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd)]>;

def ESP_ADDX4: EspVInst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.addx4\t $rd, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_ADDX4_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd),
				"!esp_addx4_p $rd, $rs1, $rs2",
			[(int_riscv_esp_addx4 GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd)]>;

def ESP_SAT: EspVInst<(outs GPRPIE:$rsdr), (ins GPRPIE:$rs0, GPRPIE:$rs1, GPRPIE:$rsd),
                   "esp.sat\t $rsd, $rs0, $rs1", []> 
{
  bits<5> rs0;
  bits<5> rs1;
  bits<5> rsd;
  bits<5> rsdr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SAT_P : PseudoEsp<(outs GPRPIE:$rsdr), (ins GPRPIE:$rs0, GPRPIE:$rs1, GPRPIE:$rsd),
				"!esp_sat_p $rsd, $rs0, $rs1",
			[(set GPRPIE:$rsdr, (int_riscv_esp_sat GPRPIE:$rs0, GPRPIE:$rs1, GPRPIE:$rsd))]>;

def ESP_SUBX2: EspVInst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.subx2\t $rd, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SUBX2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd),
				"!esp_subx2_p $rd, $rs1, $rs2",
			[(int_riscv_esp_subx2 GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd)]>;

def ESP_SUBX4: EspVInst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.subx4\t $rd, $rs1, $rs2", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SUBX4_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd),
				"!esp_subx4_p $rd, $rs1, $rs2",
			[(int_riscv_esp_subx4 GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd)]>;

def ESP_ANDQ: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.andq\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_andq_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_ANDQ_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_andq_p $qz, $qx, $qy",
			[(int_riscv_esp_andq timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_NOTQ: EspVInst<(outs QR:$qz), (ins QR:$qx),
                   "esp.notq\t $qz, $qx", 
                   [(set QR:$qz, (int_riscv_esp_notq_m QR:$qx))]> 
{
  bits<3> qx;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_NOTQ_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qz),
				"!esp_notq_p $qz, $qx",
			[(int_riscv_esp_notq timm:$qx, timm:$qz)]>;

def ESP_ORQ: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.orq\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_orq_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_ORQ_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_orq_p $qz, $qx, $qy",
			[(int_riscv_esp_orq timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_XORQ: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.xorq\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_xorq_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_XORQ_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_xorq_p $qz, $qx, $qy",
			[(int_riscv_esp_xorq timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_S16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.s16\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_EQ_S16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_S32: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.s32\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_EQ_S32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_s32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_s32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_S8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.s8\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vcmp_eq_s8_m QR:$qx, QR:$qy))]>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_EQ_S8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_U16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.u16\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_EQ_U16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_U32: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.u32\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_EQ_U32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_u32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_u32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_U8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.eq.u8\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vcmp_eq_u8_m QR:$qx, QR:$qy))]>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_EQ_U8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_u8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_S16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.s16\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_GT_S16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_S32: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.s32\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_GT_S32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_s32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_s32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_S8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.s8\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vcmp_gt_s8_m QR:$qx, QR:$qy))]>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_GT_S8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_U16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.u16\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_GT_U16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_U32: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.u32\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_GT_U32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_u32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_u32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_U8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.gt.u8\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vcmp_gt_u8_m QR:$qx, QR:$qy))]>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_GT_U8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_u8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_S16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.s16\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_LT_S16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_s16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_s16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_S32: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.s32\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_LT_S32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_s32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_s32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_S8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.s8\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vcmp_lt_s8_m QR:$qx, QR:$qy))]>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_LT_S8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_s8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_s8 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_U16: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.u16\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_LT_U16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_u16_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_u16 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_U32: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.u32\t $qz, $qx, $qy", 
                   []>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_LT_U32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_u32_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_u32 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_U8: EspVInst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "esp.vcmp.lt.u8\t $qz, $qx, $qy", 
                   [(set QR:$qz, (int_riscv_esp_vcmp_lt_u8_m QR:$qx, QR:$qy))]>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VCMP_LT_U8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_u8_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_u8 timm:$qx, timm:$qy, timm:$qz)]>;

let Defs = []<Register> in
def ESP_MOV_S16_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), (ins QR:$qu),
                   "esp.mov.s16.qacc\t $qu", 
                   []> 
{
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MOV_S16_QACC_P : PseudoEsp<(outs), (ins imm8:$qu),
				"!esp_mov_s16_qacc_p $qu",
			[(int_riscv_esp_mov_s16_qacc timm:$qu)]>;

let Defs = []<Register> in
def ESP_MOV_S8_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), (ins QR:$qu),
                   "esp.mov.s8.qacc\t $qu", 
                   []> 
{
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MOV_S8_QACC_P : PseudoEsp<(outs), (ins imm8:$qu),
				"!esp_mov_s8_qacc_p $qu",
			[(int_riscv_esp_mov_s8_qacc timm:$qu)]>;

let Defs = []<Register> in
def ESP_MOV_U16_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), (ins QR:$qu),
                   "esp.mov.u16.qacc\t $qu", 
                   []> 
{
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MOV_U16_QACC_P : PseudoEsp<(outs), (ins imm8:$qu),
				"!esp_mov_u16_qacc_p $qu",
			[(int_riscv_esp_mov_u16_qacc timm:$qu)]>;

let Defs = []<Register> in
def ESP_MOV_U8_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), (ins QR:$qu),
                   "esp.mov.u8.qacc\t $qu", 
                   []> 
{
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MOV_U8_QACC_P : PseudoEsp<(outs), (ins imm8:$qu),
				"!esp_mov_u8_qacc_p $qu",
			[(int_riscv_esp_mov_u8_qacc timm:$qu)]>;



def ESP_MOVI_16_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qy, select_16:$sel16),
                   "esp.movi.16.a\t $qy, $rd, $sel16", 
                   []> 
{
  bits<3> qy;
  bits<4> sel16;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MOVI_16_A_P : PseudoEsp<(outs), (ins imm8:$qy, select_16:$sel16, GPRPIE:$rd),
				"!esp_movi_16_a_p $qy, $rd, $sel16",
			[(int_riscv_esp_movi_16_a timm:$qy, timm:$sel16, GPRPIE:$rd)]>;

def ESP_MOVI_16_Q: EspVInst<(outs QR:$qy), (ins QR:$qy_in, GPRPIE:$rs1, select_16:$sel16),
                   "esp.movi.16.q\t $qy, $rs1, $sel16", 
                   // Intrinsic signature: (v8i16 qy_in, i32 sel16, i32 rs1) -> v8i16
                   // Instruction: ESP.MOVI.16.Q qy, rs1, sel16
                   // Parameter order in pattern: (qy_in, sel16, rs1) matches intrinsic signature
                   // Instruction uses: (qy_in, rs1, sel16) - note: rs1 and sel16 are swapped in instruction
                   []> 
{
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Add constraint: output register is the same as the input register
  let Constraints = "$qy = $qy_in";
  
  // Keep existing bit encoding
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
def ESP_MOVI_16_Q_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, select_16:$sel16, imm8:$qy),
				"!esp_movi_16_q_p $qy, $rs1, $sel16",
			[(int_riscv_esp_movi_16_q GPRPIE:$rs1, timm:$sel16, timm:$qy)]>;

def ESP_MOVI_32_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qy, select_4:$sel4),
                   "esp.movi.32.a\t $qy, $rd, $sel4", 
                   []> 
{
  bits<3> qy;
  bits<2> sel4;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MOVI_32_A_P : PseudoEsp<(outs), (ins imm8:$qy, select_4:$sel4, GPRPIE:$rd),
				"!esp_movi_32_a_p $qy, $rd, $sel4",
			[(int_riscv_esp_movi_32_a timm:$qy, timm:$sel4, GPRPIE:$rd)]>;

def ESP_MOVI_32_Q: EspVInst<(outs QR:$qy), (ins QR:$qy_in, GPRPIE:$rs1, select_4:$sel4),
                   "esp.movi.32.q\t $qy, $rs1, $sel4", 
                   // Intrinsic signature: (v4i32 qy_in, i32 sel4, i32 rs1) -> v4i32
                   // Instruction: ESP.MOVI.32.Q qy, rs1, sel4
                   // Parameter order in pattern: (qy_in, sel4, rs1) matches intrinsic signature
                   // Instruction uses: (qy_in, rs1, sel4) - note: rs1 and sel4 are swapped in instruction
                   []> 
{
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  let Constraints = "$qy = $qy_in";
  
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

// Note: INSERT_SUBREG patterns are not needed for MOVI.32.Q because:
// 1. MOVI.32.Q is already matched via intrinsic (int_riscv_esp_movi_32_q_m)
// 2. INSERT_SUBREG is a machine instruction node, not a SelectionDAG node that can be matched
// 3. If needed, insertelement operations should be lowered to MOVI.32.Q via Custom Lowering
//    in RISCVISelLowering.cpp or RISCVESPVISelLowering.cpp

let usesCustomInserter = 1 in 
def ESP_MOVI_32_Q_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, select_4:$sel4, imm8:$qy),
				"!esp_movi_32_q_p $qy, $rs1, $sel4",
			[(int_riscv_esp_movi_32_q GPRPIE:$rs1, timm:$sel4, timm:$qy)]>;

def ESP_MOVI_8_A: EspVInst<(outs GPRPIE:$rd), (ins QR:$qy, select_16:$sel16),
                   "esp.movi.8.a\t $qy, $rd, $sel16", 
                   []> 
{
  bits<3> qy;
  bits<4> sel16;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_MOVI_8_A_P : PseudoEsp<(outs), (ins imm8:$qy, select_16:$sel16, GPRPIE:$rd),
				"!esp_movi_8_a_p $qy, $rd, $sel16",
			[(int_riscv_esp_movi_8_a timm:$qy, timm:$sel16, GPRPIE:$rd)]>;

// Patterns for ESP_MOVI_8_A with subregister support
// Note: Extract from v16i8 (QR) is already matched via intrinsic in instruction definition
// 
// LIMITATION: TableGen cannot handle nested extract_subvector + extractelt patterns
// TableGen's type inference system fails with "Type set is empty" error when trying to
// match patterns like: (i32 (zext (extractelt (extract_subvector (v16i8 QR:$qy), ...), ...)))
// 
// This is a known TableGen limitation. The subregister extraction from v8i8 vectors
// should be handled via Custom Lowering in RISCVISelLowering.cpp or RISCVESPVISelLowering.cpp
// instead of TableGen patterns.
//
// TODO: Implement Custom Lowering for extractelt from extract_subvector to ESP_MOVI_8_A
// The Custom Lowering should:
// 1. Detect extractelt from extract_subvector (v16i8 -> v8i8) patterns
// 2. Calculate the correct sel16 value based on extract_subvector index and extractelt index
// 3. Generate ESP_MOVI_8_A instruction with correct sel16
//
// Example Custom Lowering logic:
// - extract_subvector(v16i8, 0) + extractelt(0-7) -> ESP_MOVI_8_A(sel16 = 0-7)
// - extract_subvector(v16i8, 8) + extractelt(0-7) -> ESP_MOVI_8_A(sel16 = 8-15)

def ESP_MOVI_8_Q: EspVInst<(outs QR:$qy), (ins QR:$qy_in, GPRPIE:$rs1, select_16:$sel16),
                   "esp.movi.8.q\t $qy, $rs1, $sel16", 
                   // Intrinsic signature: (v16i8 qy_in, i32 sel16, i32 rs1) -> v16i8
                   // Instruction: ESP.MOVI.8.Q qy, rs1, sel16
                   // Parameter order in pattern: (qy_in, sel16, rs1) matches intrinsic signature
                   // Instruction uses: (qy_in, rs1, sel16) - note: rs1 and sel16 are swapped in instruction
                   []> 
{
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qy;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  let Constraints = "$qy = $qy_in";
  
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

def ESP_MOVI_8_Q_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, select_16:$sel16, imm8:$qy),
				"!esp_movi_8_q_p $qy, $rs1, $sel16",
			[(int_riscv_esp_movi_8_q GPRPIE:$rs1, timm:$sel16, timm:$qy)]>;

def ESP_MOVX_R_CFG: EspVInst<(outs GPRPIE:$rd), (ins), // 
                   "esp.movx.r.cfg\t $rd",
                   // Pattern: set rd = intrinsic()
                   []>
{
  bits<5> rd;
  let hasSideEffects = 1;  
  let mayLoad = 0;
  let mayStore = 0;
  
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

// ESP.MOVX.R.FFT.BIT.WIDTH reads from FFT_BIT_WIDTH register (FFT_BIT_WIDTH is input source)
// For explicit state passing, FFT_BIT_WIDTH (32-bit) is passed as explicit parameter in intrinsic
// Instruction: rd[31:0] = {28'b0, FFT_BIT_WIDTH[3:0]} (hardware extracts only low 4 bits from FFT_BIT_WIDTH)
def ESP_MOVX_R_FFT_BIT_WIDTH: EspVInst<(outs GPRPIE:$rd), (ins FFT_BIT_WIDTHReg:$fft_bit_width_in),
                   "esp.movx.r.fft.bit.width\t $rd", 
                   // FFT_BIT_WIDTH is 32-bit, but instruction outputs only 32-bit rd
                   // Hardware automatically extracts FFT_BIT_WIDTH[3:0] and zero-extends to 32-bit
                   // Intrinsic returns i32 directly, no trunc needed
                   [(set GPRPIE:$rd, (int_riscv_esp_movx_r_fft_bit_width_m FFT_BIT_WIDTHReg:$fft_bit_width_in))]> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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

let Uses = [PERF] in
def ESP_MOVX_R_PERF: EspVInst<(outs GPRPIE:$rd), (ins PERFReg:$perf_in, GPRPIE:$rs1),
                   "esp.movx.r.perf\t $rd, $rs1",
                   // PERF is explicit in intrinsic and instruction (explicit state passing)
                   []>
{
  bits<5> rs1;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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

// ESP.MOVX.R.SAR reads from SAR register (SAR is input source)
// For explicit state passing, SAR (64-bit) is passed as explicit parameter in intrinsic
// Instruction: rd[31:0] = {26'b0, SAR[5:0]} (hardware extracts only low 6 bits from SAR)
def ESP_MOVX_R_SAR: EspVInst<(outs GPRPIE:$rd), (ins SARReg:$sar_in),
                   "esp.movx.r.sar\t $rd", 
                   // SAR is 64-bit, but instruction outputs only 32-bit rd
                   // Hardware automatically extracts SAR[5:0] and zero-extends to 32-bit
                   // Intrinsic returns i32 directly, no trunc needed
                   [(set GPRPIE:$rd, (int_riscv_esp_movx_r_sar_m SARReg:$sar_in))]> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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

// ESP.MOVX.R.SAR_BYTES reads from SAR_BYTES register (SAR_BYTES is input source)
// For explicit state passing, SAR_BYTES (64-bit) is passed as explicit parameter in intrinsic
// Instruction: rd[31:0] = {28'b0, SAR_BYTES[3:0]} (hardware extracts only low 4 bits from SAR_BYTES)
def ESP_MOVX_R_SAR_BYTES: EspVInst<(outs GPRPIE:$rd), (ins SAR_BYTESReg:$sar_bytes_in),
                   "esp.movx.r.sar.bytes\t $rd", 
                   // SAR_BYTES is 64-bit, but instruction outputs only 32-bit rd
                   // Hardware automatically extracts SAR_BYTES[3:0] and zero-extends to 32-bit
                   // Intrinsic returns i32 directly, no trunc needed
                   [(set GPRPIE:$rd, (int_riscv_esp_movx_r_sar_bytes_m SAR_BYTESReg:$sar_bytes_in))]> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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

def ESP_MOVX_R_XACC_H: EspVInst<(outs GPRPIE:$rd), (ins XACC_HIGH:$xacc_h_in),
                   "esp.movx.r.xacc.h\t $rd", 
                   // Subregister model: reads from XACC[39:32] (high 8 bits)
                   // Return value: XACC[39:32] (i32, zero-extended from 8-bit)
                   // Input: xacc_h_in (passthru for explicit state passing, XACC_HIGH type to match ESP_MOVX_W_XACC_H output)
                   // Hardware reads directly from XACC register, but xacc_h_in maintains data flow dependency
                   // Pattern matching is done in RISCVInstrInfoESP32P4Patterns.td
                   []> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;  // Explicit state passing, no side effects
  
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

def ESP_MOVX_R_XACC_L: EspVInst<(outs GPRPIE:$rd), (ins XACC_LOW:$xacc_l_in),
                   "esp.movx.r.xacc.l\t $rd", 
                   // Subregister model: reads from XACC[31:0] (low 32 bits)
                   // Return value: XACC[31:0] (i32)
                   // Input: xacc_l_in (passthru for explicit state passing, XACC_LOW type to match ESP_MOVX_W_XACC_L output)
                   // Hardware reads directly from XACC register, but xacc_l_in maintains data flow dependency
                   // Pattern matching is done in RISCVInstrInfoESP32P4Patterns.td
                   []> 
{
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;  // Explicit state passing, no side effects
  
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

def ESP_MOVX_W_CFG: EspVInst<(outs), (ins GPRPIE:$rs1), 
                   "esp.movx.w.cfg\t $rs1",
                   // Pattern: intrinsic(rs1)
                   []>
{
  bits<5> rs1;
  let hasSideEffects = 1; // Must be enabled to prevent write operations from being deleted or reordered
  let mayLoad = 0;
  let mayStore = 0;
  
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

def ESP_MOVX_W_FFT_BIT_WIDTH: EspVInst<(outs FFT_BIT_WIDTHReg:$fft_bit_width_out), (ins GPRPIE:$rs1),
                   "esp.movx.w.fft.bit.width\t $rs1", 
                   // Note: FFT_BIT_WIDTH is explicit in intrinsic (returns FFT_BIT_WIDTH value, 32-bit)
                   // Instruction: FFT_BIT_WIDTH[3:0] = rs1[3:0] (hardware automatically uses only low 4 bits of rs1)
                   // Return value: FFT_BIT_WIDTH (32-bit, zero-extended from rs1[3:0])
                   // Hardware automatically handles the 4-bit copy, no compiler masking needed
                   [(set FFT_BIT_WIDTHReg:$fft_bit_width_out, (int_riscv_esp_movx_w_fft_bit_width_m GPRPIE:$rs1))]>
{
  // Return value is FFT_BIT_WIDTH (32-bit), zero-extended from rs1[3:0]
  // Hardware automatically writes FFT_BIT_WIDTH[3:0] = rs1[3:0] (only low 4 bits are used)
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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

let Defs = [PERF] in
def ESP_MOVX_W_PERF: EspVInst<(outs PERFReg:$perf_out), (ins PERFReg:$perf_in, GPRPIE:$rs1),
                   "esp.movx.w.perf\t $rs1",
                   // PERF is explicit in intrinsic and instruction (explicit state passing)
                   []>
{
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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

def ESP_MOVX_W_SAR: EspVInst<(outs SARReg:$sar_out), (ins GPRPIE:$rs1),
                   "esp.movx.w.sar\t $rs1", 
                   // Note: SAR is explicit in intrinsic (returns SAR value, 32-bit)
                   // Instruction: SAR[5:0] = rs1[5:0] (hardware automatically uses only low 6 bits of rs1)
                   // Return value: SAR (32-bit, zero-extended from rs1[5:0])
                   // Hardware automatically handles the 6-bit copy, no compiler masking needed
                   [(set SARReg:$sar_out, (int_riscv_esp_movx_w_sar_m GPRPIE:$rs1))]>
{
  // Return value is SAR (64-bit), zero-extended from rs1[5:0]
  // Hardware automatically writes SAR[5:0] = rs1[5:0] (only low 6 bits are used)
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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

def ESP_MOVX_W_SAR_BYTES: EspVInst<(outs SAR_BYTESReg:$sar_bytes_out), (ins GPRPIE:$rs1),
                   "esp.movx.w.sar.bytes\t $rs1", 
                   // Note: SAR_BYTES is explicit in intrinsic (returns SAR value, 64-bit)
                   // Instruction: SAR_BYTES[3:0] = rs1[3:0] (hardware automatically uses only low 4 bits of rs1)
                   // Return value: SAR_BYTES (64-bit, zero-extended from rs1[3:0])
                   // Hardware automatically handles the 4-bit copy, no compiler masking needed
                   [(set SAR_BYTESReg:$sar_bytes_out, (int_riscv_esp_movx_w_sar_bytes_m GPRPIE:$rs1))]>
{
  // Return value is SAR_BYTES (64-bit), zero-extended from rs1[3:0]
  // Hardware automatically writes SAR_BYTES[3:0] = rs1[3:0] (only low 4 bits are used)
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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

def ESP_MOVX_W_XACC_H: EspVInst<(outs XACC_HIGH:$xacc_out), (ins GPRPIE:$rs1),
                   "esp.movx.w.xacc.h\t $rs1", 
                   // Subregister model: writes to XACC[39:32] (high 8 bits)
                   // Instruction writes XACC[39:32] = rs1[7:0], XACC[31:0] remains unchanged (hardware passthru)
                   // Returns XACC_HIGH register (matches ESP_MOVX_R_XACC_H input type)
                   // Pattern matching is done in RISCVInstrInfoESP32P4Patterns.td
                   []> 
{
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;  // Explicit state passing, no side effects
  
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

def ESP_MOVX_W_XACC_L: EspVInst<(outs XACC_LOW:$xacc_out), (ins GPRPIE:$rs1),
                   "esp.movx.w.xacc.l\t $rs1", 
                   // Subregister model: writes to XACC[31:0] (low 32 bits)
                   // Instruction writes XACC[31:0] = rs1[31:0], XACC[39:32] remains unchanged (hardware passthru)
                   // Returns XACC[31:0] (i32)
                   // Pattern matching is done in RISCVInstrInfoESP32P4Patterns.td
                   []> 
{
  bits<5> rs1;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;  // Explicit state passing, no side effects
  
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

def ESP_VEXT_S16: EspVInst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
                   "esp.vext.s16\t $qz, $qv, $qw", 
                   [(set QR:$qz, QR:$qv, (int_riscv_esp_vext_s16_m QR:$qw))]> 
{
  bits<3> qw;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VEXT_S16_P : PseudoEsp<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_s16_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_s16 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VEXT_S8: EspVInst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
                   "esp.vext.s8\t $qz, $qv, $qw", 
                   [(set QR:$qz, QR:$qv, (int_riscv_esp_vext_s8_m QR:$qw))]> 
{
  bits<3> qw;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VEXT_S8_P : PseudoEsp<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_s8_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_s8 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VEXT_U16: EspVInst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
                   "esp.vext.u16\t $qz, $qv, $qw", 
                   [(set QR:$qz, QR:$qv, (int_riscv_esp_vext_u16_m QR:$qw))]> 
{
  bits<3> qw;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VEXT_U16_P : PseudoEsp<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_u16_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_u16 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VEXT_U8: EspVInst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
                   "esp.vext.u8\t $qz, $qv, $qw", 
                   [(set QR:$qz, QR:$qv, (int_riscv_esp_vext_u8_m QR:$qw))]> 
{
  bits<3> qw;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VEXT_U8_P : PseudoEsp<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_u8_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_u8 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VUNZIP_16: EspVInst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vunzip.16\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VUNZIP_16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vunzip_16_p $qx, $qy",
			[(int_riscv_esp_vunzip_16 timm:$qx, timm:$qy)]>;

def ESP_VUNZIP_32: EspVInst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vunzip.32\t $qx, $qy", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VUNZIP_32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vunzip_32_p $qx, $qy",
			[(int_riscv_esp_vunzip_32 timm:$qx, timm:$qy)]>;

def ESP_VUNZIP_8: EspVInst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vunzip.8\t $qx, $qy", 
                   [(set QR:$qxr, QR:$qyr, (int_riscv_esp_vunzip_8_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VUNZIP_8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vunzip_8_p $qx, $qy",
			[(int_riscv_esp_vunzip_8 timm:$qx, timm:$qy)]>;

def ESP_VUNZIPT_16: EspVInst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
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
  let hasSideEffects = 0;
  
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
def ESP_VUNZIPT_16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vunzipt_16_p $qx, $qy, $qw",
			[(int_riscv_esp_vunzipt_16 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_VUNZIPT_8: EspVInst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
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
  let hasSideEffects = 0;
  
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
def ESP_VUNZIPT_8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vunzipt_8_p $qx, $qy, $qw",
			[(int_riscv_esp_vunzipt_8 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_VZIP_16: EspVInst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vzip.16\t $qx, $qy", 
                   [(set QR:$qxr, QR:$qyr, (int_riscv_esp_vzip_16_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VZIP_16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vzip_16_p $qx, $qy",
			[(int_riscv_esp_vzip_16 timm:$qx, timm:$qy)]>;

def ESP_VZIP_32: EspVInst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vzip.32\t $qx, $qy", 
                   [(set QR:$qxr, QR:$qyr, (int_riscv_esp_vzip_32_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VZIP_32_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vzip_32_p $qx, $qy",
			[(int_riscv_esp_vzip_32 timm:$qx, timm:$qy)]>;

def ESP_VZIP_8: EspVInst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
                   "esp.vzip.8\t $qx, $qy", 
                   [(set QR:$qxr, QR:$qyr, (int_riscv_esp_vzip_8_m QR:$qx, QR:$qy))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qxr;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VZIP_8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vzip_8_p $qx, $qy",
			[(int_riscv_esp_vzip_8 timm:$qx, timm:$qy)]>;

def ESP_VZIPT_16: EspVInst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
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
  let hasSideEffects = 0;
  
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
def ESP_VZIPT_16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vzipt_16_p $qx, $qy, $qw",
			[(int_riscv_esp_vzipt_16 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_VZIPT_8: EspVInst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
                   "esp.vzipt.8\t $qx, $qy, $qw", 
                   [(set QR:$qxr, QR:$qyr, QR:$qwr, (int_riscv_esp_vzipt_8_m QR:$qx, QR:$qy, QR:$qw))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<3> qxr;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VZIPT_8_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vzipt_8_p $qx, $qy, $qw",
			[(int_riscv_esp_vzipt_8 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_ZERO_Q: EspVInst<(outs QR:$qz), (ins),
                   "esp.zero.q\t $qz", 
                   [(set QR:$qz, (int_riscv_esp_zero_q_m))]> 
{
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_ZERO_Q_P : PseudoEsp<(outs), (ins imm8:$qz),
				"!esp_zero_q_p $qz",
			[(int_riscv_esp_zero_q timm:$qz)]>;

// ESP.ZERO.QACC - Zero entire QACC accumulator (affects both H and L)
// Output 4x128-bit subregisters for precise modeling, consistent with ESP_MOV_S16_QACC and other QACC instructions.
// This enables precise data flow tracking through phantom operands.
// 
// CRITICAL: Defs = []<Register> to eliminate implicit-def and prevent self-interference.
// Without this, RA sees implicit-def $qacc_h_reg/$qacc_l_reg and cannot allocate
// the virtual registers to the same physical registers, causing spill attempts.
//
// Rematerialization marks: Tell RA that if spilling is needed, delete me and regenerate me when needed.
// This prevents storeRegToStackSlot crash since RA will rematerialize instead of spilling.
let Defs = []<Register>,        // Force clear implicit-def to prevent self-interference
    isReMaterializable = 1,      // Tell RA: if spilling is needed, delete me and regenerate me when needed
    isAsCheapAsAMove = 1,       // Tell RA: regenerating me is as cheap as a Move
    hasSideEffects = 0          // Tell RA: I don't read/write memory, don't modify other state, safe to delete/rebuild
in
def ESP_ZERO_QACC: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h), (ins),
                   "esp.zero.qacc\t", 
                   [(set QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, (int_riscv_esp_zero_qacc_m))]> 
{
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_ZERO_QACC_P : PseudoEsp<(outs), (ins),
				"!esp_zero_qacc_p",
			[(int_riscv_esp_zero_qacc)]>;

// ESP_ZERO_XACC: Only explicit output XACCReg:$xacc, no implicit Defs
// This avoids double constraint: explicit VReg output will be allocated to physical XACC
// by register allocator, achieving the same effect without conflicting constraints
// Note: Pattern matches SDNode instead of intrinsic directly to support Chain/Glue
//
// CRITICAL: Defs = []<Register> to eliminate implicit-def and prevent self-interference.
// Without this, RA might see implicit-def $xacc and cannot allocate the virtual register
// to the same physical register, causing spill attempts.
//
// Rematerialization marks: Tell RA that if spilling is needed, delete me and regenerate me when needed.
let Defs = []<Register>,        // Force clear implicit-def to prevent self-interference
    isReMaterializable = 1,      // Tell RA: if spilling is needed, delete me and regenerate me when needed
    isAsCheapAsAMove = 1,        // Tell RA: regenerating me is as cheap as a Move
    hasSideEffects = 0           // Tell RA: I don't read/write memory, don't modify other state, safe to delete/rebuild
in
// Mixed model: XACC as {i32 low, i32 high}
// Hardware instruction operates on implicit XACC register, but outputs are explicit for data flow
// Pattern matching: SDNode returns {i32, i32}, instruction outputs XACC_LOW and XACC_HIGH
let Defs = [XACC] in
def ESP_ZERO_XACC: EspVInst<(outs XACC_LOW:$xacc_low, XACC_HIGH:$xacc_high), (ins),
                   "esp.zero.xacc\t", 
                   []>
{
  let mayStore = 0;
  let mayLoad = 0;
  
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
def ESP_ZERO_XACC_P : PseudoEsp<(outs), (ins),
				"!esp_zero_xacc_p",
			[(int_riscv_esp_zero_xacc)]>;

// ESP.FFT.AMS.S16.LD.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_FFT_AMS_S16_LD_INCP: EspVInst<(outs QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2, SARReg:$sar_in),
                   "esp.fft.ams.s16.ld.incp\t $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2", 
                   [(set QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r, (riscv_esp_fft_ams_s16_ld_incp_m QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
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
def ESP_FFT_AMS_S16_LD_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qw, GPRPIE:$rs1, select_2:$sel2, imm8:$qu, imm8:$qz, imm8:$qv),
				"!esp_fft_ams_s16_ld_incp_p $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_ams_s16_ld_incp timm:$qx, timm:$qy, timm:$qw, GPRPIE:$rs1, timm:$sel2, timm:$qu, timm:$qz, timm:$qv))]>;

// ESP.FFT.AMS.S16.LD.INCP.UAUP with explicit phantom operands
// UA_STATE is input/output (phantom operand), SAR_BYTES and SAR are input-only (phantom operands)
// Phantom operands appear in (ins) and (outs) for data flow dependency,
// but are NOT printed in the assembly string (only $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2 are printed)
// Note: No Defs/Uses = [UA_STATE, SAR_BYTES, SAR] because we use explicit phantom operands
// UA_STATE: input and output (updated by instruction)
// SAR_BYTES: input only (read-only, used for shift amount calculation)
// SAR: input only (read-only, used for shift operations)
def ESP_FFT_AMS_S16_LD_INCP_UAUP: EspVInst<(outs QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r, UA_STATEReg:$ua_state_out), 
                                            (ins QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2, UA_STATEReg:$ua_state_in, SAR_BYTESReg:$sar_bytes_in, SARReg:$sar_in),
                   "esp.fft.ams.s16.ld.incp.uaup\t $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2", 
                   [(set QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r, UA_STATEReg:$ua_state_out, 
                       (riscv_esp_fft_ams_s16_ld_incp_uaup_m QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2, UA_STATEReg:$ua_state_in, SAR_BYTESReg:$sar_bytes_in, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // UA_STATE tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $ua_state_out = $ua_state_in";
  
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
def ESP_FFT_AMS_S16_LD_INCP_UAUP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qw, GPRPIE:$rs1, select_2:$sel2, imm8:$qu, imm8:$qz, imm8:$qv),
				"!esp_fft_ams_s16_ld_incp_uaup_p $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_ams_s16_ld_incp_uaup timm:$qx, timm:$qy, timm:$qw, GPRPIE:$rs1, timm:$sel2, timm:$qu, timm:$qz, timm:$qv))]>;

// ESP.FFT.AMS.S16.LD.R32.DECP - Same as LD.INCP but decrements pointer instead of incrementing
// SAR is passed explicitly as SARReg:$sar_in (phantom operand) for explicit state passing
// Note: No Defs/Uses = [SAR] because we use explicit phantom operand
def ESP_FFT_AMS_S16_LD_R32_DECP: EspVInst<(outs QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2, SARReg:$sar_in),
                   "esp.fft.ams.s16.ld.r32.decp\t $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2",
                   [(set QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r, (riscv_esp_fft_ams_s16_ld_r32_decp_m QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
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
def ESP_FFT_AMS_S16_LD_R32_DECP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qw, GPRPIE:$rs1, select_2:$sel2, imm8:$qu, imm8:$qz, imm8:$qv),
				"!esp_fft_ams_s16_ld_r32_decp_p $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_ams_s16_ld_r32_decp timm:$qx, timm:$qy, timm:$qw, GPRPIE:$rs1, timm:$sel2, timm:$qu, timm:$qz, timm:$qv))]>;

// ESP.FFT.AMS.S16.ST.INCP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_FFT_AMS_S16_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r, GPRPIE:$rs2r), (ins QR:$qx, QR:$qy, QR:$qw, QR:$qu, GPRPIE:$rs1, GPRPIE:$rs2, select_2:$sel2, SARReg:$sar_in),
                   "esp.fft.ams.s16.st.incp\t $qu, $qz, $rs2, $rs1, $qx, $qw, $qy, $sel2",
                   [(set QR:$qz, GPRPIE:$rs1r, (riscv_esp_fft_ams_s16_st_incp_m QR:$qx, QR:$qy, QR:$qw, QR:$qu, GPRPIE:$rs1, GPRPIE:$rs2, select_2:$sel2, SARReg:$sar_in)),
                    (set GPRPIE:$rs2r, GPRPIE:$rs2)]> 
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
  let hasSideEffects = 0;
  
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
def ESP_FFT_AMS_S16_ST_INCP_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw, imm8:$qu, GPRPIE:$rs1, GPRPIE:$rs2, select_2:$sel2, imm8:$qz),
				"!esp_fft_ams_s16_st_incp_p $qu, $qz, $rs2, $rs1, $qx, $qw, $qy, $sel2",
			[(int_riscv_esp_fft_ams_s16_st_incp timm:$qx, timm:$qy, timm:$qw, timm:$qu, GPRPIE:$rs1, GPRPIE:$rs2, timm:$sel2, timm:$qz)]>;

// ESP.FFT.BITREV with explicit FFT_BIT_WIDTH state passing
// FFT_BIT_WIDTH is passed explicitly as FFT_BIT_WIDTHReg:$fft_bit_width_in (32-bit) for explicit state passing
// Note: No Uses = [FFT_BIT_WIDTH] because FFT_BIT_WIDTH is explicit operand
def ESP_FFT_BITREV: EspVInst<(outs GPRPIE:$rs1r, QR:$qvr), (ins GPRPIE:$rs1, FFT_BIT_WIDTHReg:$fft_bit_width_in),
                   "esp.fft.bitrev\t $qvr, $rs1",
                   // Note: FFT_BIT_WIDTH is explicit in intrinsic (passed as i32 parameter)
                   // Instruction uses FFT_BIT_WIDTH[3:0] for bit reversal operation
                   // Pattern: use i32 instead of FFT_BIT_WIDTHReg to match SDNode operand type
                   // LLVM will insert COPY instruction to convert GPR to FFT_BIT_WIDTHReg
                   [(set GPRPIE:$rs1r, QR:$qvr, (riscv_esp_fft_bitrev_m GPRPIE:$rs1, i32:$fft_bit_width_in))]> 
{
  bits<5> rs1;
  bits<5> rs1r;
  bits<3> qvr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qvr{2-0};
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
def ESP_FFT_BITREV_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, imm8:$qv),
				"!esp_fft_bitrev_p $qv, $rs1",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_bitrev GPRPIE:$rs1, timm:$qv))]>;

// ESP.FFT.CMUL.S16.LD.XP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_FFT_CMUL_S16_LD_XP: EspVInst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, select_8:$sel8, SARReg:$sar_in),
                   "esp.fft.cmul.s16.ld.xp\t $qu, $rs1, $rs2, $qz, $qy, $qx, $sel8",
                   [(set QR:$qz, QR:$qu, GPRPIE:$rs1r, (riscv_esp_fft_cmul_s16_ld_xp_m QR:$qx, QR:$qy, GPRPIE:$rs1, GPRPIE:$rs2, select_8:$sel8, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
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
def ESP_FFT_CMUL_S16_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_8:$sel8, imm8:$qz, imm8:$qu),
				"!esp_fft_cmul_s16_ld_xp_p $qu, $rs1, $rs2, $qz, $qy, $qx, $sel8",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_cmul_s16_ld_xp GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel8, timm:$qz, timm:$qu))]>;

// ESP.FFT.CMUL.S16.ST.XP with explicit SAR state passing
// SAR is passed explicitly as SARReg:$sar_in (64-bit) for explicit state passing
def ESP_FFT_CMUL_S16_ST_XP: EspVInst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, select_4:$sel4, select_4:$upd4, select_8:$sel8, SARReg:$sar_in),
                   "esp.fft.cmul.s16.st.xp\t $qy, $qx, $qu, $rs1, $rs2, $sel8, $upd4, $sel4",
                   [(set GPRPIE:$rs1r, (riscv_esp_fft_cmul_s16_st_xp_m QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, GPRPIE:$rs2, select_8:$sel8, select_4:$upd4, select_4:$sel4, SARReg:$sar_in))]> 
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
  let hasSideEffects = 0;
  
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
def ESP_FFT_CMUL_S16_ST_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, select_4:$sel4, select_4:$upd4, select_8:$sel8),
				"!esp_fft_cmul_s16_st_xp_p $qy, $qx, $qu, $rs1, $rs2, $sel8, $upd4, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_cmul_s16_st_xp GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sel4, timm:$upd4, timm:$sel8))]>;

def ESP_FFT_R2BF_S16: EspVInst<(outs QR:$qz, QR:$qv), (ins QR:$qx, QR:$qy, select_2:$sel2),
                   "esp.fft.r2bf.s16\t $qz, $qv, $qx, $qy, $sel2",
                   [(set QR:$qz, QR:$qv, (riscv_esp_fft_r2bf_s16_m QR:$qx, QR:$qy, select_2:$sel2))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sel2;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_FFT_R2BF_S16_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, select_2:$sel2, imm8:$qz, imm8:$qv),
				"!esp_fft_r2bf_s16_p $qz, $qv, $qx, $qy, $sel2",
			[(int_riscv_esp_fft_r2bf_s16 timm:$qx, timm:$qy, timm:$sel2, timm:$qz, timm:$qv)]>;

def ESP_FFT_R2BF_S16_ST_INCP: EspVInst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4),
                   "esp.fft.r2bf.s16.st.incp\t $qz, $qx, $qy, $rs1, $sel4",
                   [(set QR:$qz, GPRPIE:$rs1r, (riscv_esp_fft_r2bf_s16_st_incp_m QR:$qx, QR:$qy, GPRPIE:$rs1, select_4:$sel4))]> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_FFT_R2BF_S16_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, select_4:$sel4, imm8:$qz),
				"!esp_fft_r2bf_s16_st_incp_p $qz, $qx, $qy, $rs1, $sel4",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_r2bf_s16_st_incp timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sel4, timm:$qz))]>;

def ESP_FFT_VST_R32_DECP: EspVInst<(outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, select_2:$sel2),
                   "esp.fft.vst.r32.decp\t $qu, $rs1, $sel2",
                   [(set GPRPIE:$rs1r, (riscv_esp_fft_vst_r32_decp_m QR:$qu, GPRPIE:$rs1, select_2:$sel2))]> 
{
  bits<3> qu;
  bits<5> rs1;
  bits<1> sel2;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_FFT_VST_R32_DECP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, select_2:$sel2),
				"!esp_fft_vst_r32_decp_p $qu, $rs1, $sel2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_vst_r32_decp timm:$qu, GPRPIE:$rs1, timm:$sel2))]>;

// ESP.LD.128.USAR.IP with explicit SAR_BYTES output
// SAR_BYTES is explicit output for explicit state passing
def ESP_LD_128_USAR_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, SAR_BYTESReg:$sar_bytes_out), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.128.usar.ip\t $qu, $rs1, $off25616", 
                   // Note: SAR_BYTES[3:0] = rs1[3:0] (hardware automatically extracts only low 4 bits from rs1)
                   [(set QR:$qu, GPRPIE:$rs1r, SAR_BYTESReg:$sar_bytes_out, (riscv_esp_ld_128_usar_ip_m GPRPIE:$rs1, offset_256_16:$off25616))]> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LD_128_USAR_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616, imm8:$qu),
				"!esp_ld_128_usar_ip_p $qu, $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_128_usar_ip GPRPIE:$rs1, timm:$off25616, timm:$qu))]>;

// ESP.LD.128.USAR.XP with explicit SAR_BYTES output
// SAR_BYTES is explicit output for explicit state passing
def ESP_LD_128_USAR_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, SAR_BYTESReg:$sar_bytes_out), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ld.128.usar.xp\t $qu, $rs1, $rs2", 
                   // Note: SAR_BYTES[3:0] = rs1[3:0] (hardware automatically extracts only low 4 bits from rs1)
                   [(set QR:$qu, GPRPIE:$rs1r, SAR_BYTESReg:$sar_bytes_out, (riscv_esp_ld_128_usar_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LD_128_USAR_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_ld_128_usar_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_128_usar_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

// Mixed model: XACC as {i32 low, i32 high}
// Hardware instruction operates on implicit XACC register, but outputs are explicit for data flow
// Pattern matching: SDNode returns {ptr, i32, i32}, instruction outputs XACC_LOW and XACC_HIGH
def ESP_LD_XACC_IP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low, XACC_HIGH:$xacc_high), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.ld.xacc.ip\t $rs1, $off2568", 
                   []> 
{
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Force input and output to use same physical register
  // This is critical for singleton register classes (only one physical register)
  // Without this, register allocator may fail when it thinks it needs two registers
  let Constraints = "$rs1r = $rs1, $xacc_low = $xacc_low_in, $xacc_high = $xacc_high_in";
  
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
def ESP_LD_XACC_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_ld_xacc_ip_p $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_xacc_ip GPRPIE:$rs1, timm:$off2568))]>;

let Defs = []<Register> in
def ESP_LDQA_S16_128_IP: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ldqa.s16.128.ip\t $rs1, $off25616", 
                   []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LDQA_S16_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_s16_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s16_128_ip GPRPIE:$rs1, timm:$off25616))]>;

let Defs = []<Register> in
def ESP_LDQA_S16_128_XP: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ldqa.s16.128.xp\t $rs1, $rs2", 
                   []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LDQA_S16_128_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_s16_128_xp_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s16_128_xp GPRPIE:$rs2, GPRPIE:$rs1))]>;

let Defs = []<Register> in
def ESP_LDQA_S8_128_IP: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ldqa.s8.128.ip\t $rs1, $off25616", 
                   []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LDQA_S8_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_s8_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s8_128_ip GPRPIE:$rs1, timm:$off25616))]>;

let Defs = []<Register> in
def ESP_LDQA_S8_128_XP: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ldqa.s8.128.xp\t $rs1, $rs2", 
                   []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LDQA_S8_128_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_s8_128_xp_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s8_128_xp GPRPIE:$rs2, GPRPIE:$rs1))]>;

let Defs = []<Register> in
def ESP_LDQA_U16_128_IP: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ldqa.u16.128.ip\t $rs1, $off25616", 
                   []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LDQA_U16_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_u16_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u16_128_ip GPRPIE:$rs1, timm:$off25616))]>;

let Defs = [QACC_H_REG, QACC_L_REG] in
let Defs = []<Register> in
def ESP_LDQA_U16_128_XP: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ldqa.u16.128.xp\t $rs1, $rs2", 
                   []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LDQA_U16_128_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_u16_128_xp_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u16_128_xp GPRPIE:$rs2, GPRPIE:$rs1))]>;

let Defs = [QACC_H_REG, QACC_L_REG] in
let Defs = []<Register> in
def ESP_LDQA_U8_128_IP: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ldqa.u8.128.ip\t $rs1, $off25616", 
                   []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LDQA_U8_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_u8_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u8_128_ip GPRPIE:$rs1, timm:$off25616))]>;

let Defs = []<Register> in
def ESP_LDQA_U8_128_XP: EspVInst<(outs QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ldqa.u8.128.xp\t $rs1, $rs2", 
                   []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LDQA_U8_128_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_u8_128_xp_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u8_128_xp GPRPIE:$rs2, GPRPIE:$rs1))]>;

def ESP_VLDBC_16_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_2:$off2564),
                   "esp.vldbc.16.ip\t $qu, $rs1, $off2564", 
                   [(set QR:$qu, GPRPIE:$rs1r, (riscv_esp_vldbc_16_ip_m GPRPIE:$rs1, offset_256_2:$off2564))]> 
{
  bits<5> rs1;
  bits<8> off2564;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDBC_16_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_2:$off2564, imm8:$qu),
				"!esp_vldbc_16_ip_p $qu, $rs1, $off2564",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_16_ip GPRPIE:$rs1, timm:$off2564, timm:$qu))]>;

def ESP_VLDBC_16_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldbc.16.xp\t $qu, $rs1, $rs2", 
                   [(set QR:$qu, GPRPIE:$rs1r, (riscv_esp_vldbc_16_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDBC_16_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vldbc_16_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_16_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLDBC_32_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_4:$off2564),
                   "esp.vldbc.32.ip\t $qu, $rs1, $off2564", 
                   [(set QR:$qu, GPRPIE:$rs1r, (riscv_esp_vldbc_32_ip_m GPRPIE:$rs1, offset_256_4:$off2564))]> 
{
  bits<5> rs1;
  bits<8> off2564;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDBC_32_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_4:$off2564, imm8:$qu),
				"!esp_vldbc_32_ip_p $qu, $rs1, $off2564",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_32_ip GPRPIE:$rs1, timm:$off2564, timm:$qu))]>;

def ESP_VLDBC_32_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldbc.32.xp\t $qu, $rs1, $rs2", 
                   [(set QR:$qu, GPRPIE:$rs1r, (riscv_esp_vldbc_32_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDBC_32_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vldbc_32_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_32_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLDBC_8_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, imm8:$off2564),
                   "esp.vldbc.8.ip\t $qu, $rs1, $off2564", 
                   [(set QR:$qu, GPRPIE:$rs1r, (riscv_esp_vldbc_8_ip_m GPRPIE:$rs1, imm8:$off2564))]> 
{
  bits<5> rs1;
  bits<8> off2564;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDBC_8_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, imm8:$off2564, imm8:$qu),
				"!esp_vldbc_8_ip_p $qu, $rs1, $off2564",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_8_ip GPRPIE:$rs1, timm:$off2564, timm:$qu))]>;

def ESP_VLDBC_8_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldbc.8.xp\t $qu, $rs1, $rs2", 
                   [(set QR:$qu, GPRPIE:$rs1r, (riscv_esp_vldbc_8_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDBC_8_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vldbc_8_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_8_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLDEXT_S16_IP: EspVInst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vldext.s16.ip\t $qu, $qz, $rs1, $off1616", 
                   [(set QR:$qu, QR:$qz, GPRPIE:$rs1r, (riscv_esp_vldext_s16_ip_m GPRPIE:$rs1, offset_16_16:$off1616))]> 
{
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDEXT_S16_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_s16_ip_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s16_ip GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_S16_XP: EspVInst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldext.s16.xp\t $qu, $qz, $rs1, $rs2", 
                   [(set QR:$qu, QR:$qz, GPRPIE:$rs1r, (riscv_esp_vldext_s16_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDEXT_S16_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_s16_xp_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s16_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_S8_IP: EspVInst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vldext.s8.ip\t $qu, $qz, $rs1, $off1616", 
                   [(set QR:$qu, QR:$qz, GPRPIE:$rs1r, (riscv_esp_vldext_s8_ip_m GPRPIE:$rs1, offset_16_16:$off1616))]> 
{
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDEXT_S8_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_s8_ip_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s8_ip GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_S8_XP: EspVInst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldext.s8.xp\t $qu, $qz, $rs1, $rs2", 
                   [(set QR:$qu, QR:$qz, GPRPIE:$rs1r, (riscv_esp_vldext_s8_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDEXT_S8_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_s8_xp_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s8_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U16_IP: EspVInst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vldext.u16.ip\t $qu, $qz, $rs1, $off1616", 
                   [(set QR:$qu, QR:$qz, GPRPIE:$rs1r, (riscv_esp_vldext_u16_ip_m GPRPIE:$rs1, offset_16_16:$off1616))]> 
{
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDEXT_U16_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_u16_ip_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u16_ip GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U16_XP: EspVInst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldext.u16.xp\t $qu, $qz, $rs1, $rs2", 
                   [(set QR:$qu, QR:$qz, GPRPIE:$rs1r, (riscv_esp_vldext_u16_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDEXT_U16_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_u16_xp_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u16_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U8_IP: EspVInst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
                   "esp.vldext.u8.ip\t $qu, $qz, $rs1, $off1616", 
                   [(set QR:$qu, QR:$qz, GPRPIE:$rs1r, (riscv_esp_vldext_u8_ip_m GPRPIE:$rs1, offset_16_16:$off1616))]> 
{
  bits<5> rs1;
  bits<4> off1616;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDEXT_U8_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_u8_ip_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u8_ip GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U8_XP: EspVInst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vldext.u8.xp\t $qu, $qz, $rs1, $rs2", 
                   [(set QR:$qu, QR:$qz, GPRPIE:$rs1r, (riscv_esp_vldext_u8_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDEXT_U8_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_u8_xp_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u8_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDHBC_16_INCP: EspVInst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1),
                   "esp.vldhbc.16.incp\t $qu, $qz, $rs1", 
                   []> 
{
  bits<5> rs1;
  bits<3> qu;
  bits<3> qz;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLDHBC_16_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldhbc_16_incp_p $qu, $qz, $rs1",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldhbc_16_incp GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

// ESP_LD_QACC_H_H_128_IP: Load QACC_H[255:128] with subregister model
// Subregister model: uses QACC_H_HIGH (128-bit subregister) matching hardware behavior
// Instruction loads 128-bit data to QACC_H[255:128], QACC_H[127:0] remains unchanged
def ESP_LD_QACC_H_H_128_IP: EspVInst<(outs QACC_H_HIGH:$qacc_h_out, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.qacc.h.h.128.ip\t $rs1, $off25616", 
                   [(set QACC_H_HIGH:$qacc_h_out, GPRPIE:$rs1r, (riscv_esp_ld_qacc_h_h_128_ip_m GPRPIE:$rs1, offset_256_16:$off25616))]> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LD_QACC_H_H_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_h_h_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_h_h_128_ip GPRPIE:$rs1, timm:$off25616))]>;

// ESP_LD_QACC_H_L_128_IP: Load QACC_H[127:0] with subregister model
// Subregister model: uses QACC_H_LOW (128-bit subregister) matching hardware behavior
// Instruction loads 128-bit data to QACC_H[127:0], QACC_H[255:128] remains unchanged
def ESP_LD_QACC_H_L_128_IP: EspVInst<(outs QACC_H_LOW:$qacc_h_out, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.qacc.h.l.128.ip\t $rs1, $off25616", 
                   [(set QACC_H_LOW:$qacc_h_out, GPRPIE:$rs1r, (riscv_esp_ld_qacc_h_l_128_ip_m GPRPIE:$rs1, offset_256_16:$off25616))]> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LD_QACC_H_L_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_h_l_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_h_l_128_ip GPRPIE:$rs1, timm:$off25616))]>;

// ESP_LD_QACC_L_H_128_IP: Load QACC_L[255:128] with subregister model
// Subregister model: uses QACC_L_HIGH (128-bit subregister) matching hardware behavior
// Instruction loads 128-bit data to QACC_L[255:128], QACC_L[127:0] remains unchanged
def ESP_LD_QACC_L_H_128_IP: EspVInst<(outs QACC_L_HIGH:$qacc_l_out, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.qacc.l.h.128.ip\t $rs1, $off25616", 
                   [(set QACC_L_HIGH:$qacc_l_out, GPRPIE:$rs1r, (riscv_esp_ld_qacc_l_h_128_ip_m GPRPIE:$rs1, offset_256_16:$off25616))]> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LD_QACC_L_H_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_l_h_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_l_h_128_ip GPRPIE:$rs1, timm:$off25616))]>;

// ESP_LD_QACC_L_L_128_IP: Load QACC_L[127:0] with subregister model
// Subregister model: uses QACC_L_LOW (128-bit subregister) matching hardware behavior
// Instruction loads 128-bit data to QACC_L[127:0], QACC_L[255:128] remains unchanged
def ESP_LD_QACC_L_L_128_IP: EspVInst<(outs QACC_L_LOW:$qacc_l_out, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ld.qacc.l.l.128.ip\t $rs1, $off25616", 
                   [(set QACC_L_LOW:$qacc_l_out, GPRPIE:$rs1r, (riscv_esp_ld_qacc_l_l_128_ip_m GPRPIE:$rs1, offset_256_16:$off25616))]> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LD_QACC_L_L_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_l_l_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_l_l_128_ip GPRPIE:$rs1, timm:$off25616))]>;

// ESP.LD.UA.STATE.IP - Load 128-bit data to UA_STATE register
// Real instruction with phantom operand (ghost operand for data flow analysis)
// UA_STATEReg:$ua_state_out is explicit output for explicit state passing
// UA_STATEReg:$ua_state_in is phantom operand: appears in (ins) for data flow dependency,
// but is NOT printed in the assembly string (only $rs1 and $off25616 are printed)
def ESP_LD_UA_STATE_IP: EspVInst<(outs UA_STATEReg:$ua_state_out, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616, UA_STATEReg:$ua_state_in),
                   "esp.ld.ua.state.ip\t $rs1, $off25616", 
                   [(set UA_STATEReg:$ua_state_out, GPRPIE:$rs1r, (riscv_esp_ld_ua_state_ip_m GPRPIE:$rs1, offset_256_16:$off25616, UA_STATEReg:$ua_state_in))]> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // UA_STATE tied constraints: Input and output must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $ua_state_out = $ua_state_in";
  
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
def ESP_LD_UA_STATE_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_ua_state_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_ua_state_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDXQ_32: EspVInst<(outs QR:$qu), (ins GPRPIE:$rs1, QR:$qw, select_4:$sel4, select_8:$sel8),
                   "esp.ldxq.32\t $qu, $qw, $rs1, $sel4, $sel8", 
                   []> 
{
  bits<5> rs1;
  bits<3> qw;
  bits<2> sel4;
  bits<3> sel8;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_LDXQ_32_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qw, select_4:$sel4, select_8:$sel8, imm8:$qu),
				"!esp_ldxq_32_p $qu, $qw, $rs1, $sel4, $sel8",
			[(int_riscv_esp_ldxq_32 GPRPIE:$rs1, timm:$qw, timm:$sel4, timm:$sel8, timm:$qu)]>;

// Note: INSERT_SUBREG patterns for LDXQ.32 are not straightforward because
// LDXQ.32 requires $qw and $sel8 for address calculation (vaddr = rs1 + (qw[sel8] << 2)).
// INSERT_SUBREG patterns typically match simple insert operations without address calculation.
// For LDXQ.32, the intrinsic matching approach is more appropriate.
// 
// For simple insert operations (from register), use MOVI.32.Q instructions instead.
// LDXQ.32 hardware instruction already supports in-place update semantics (Read-Modify-Write),
// so no separate Two-Address variant is needed.

// ESP_ST_QACC_H_H_128_IP: Store QACC_H[255:128] to memory
// Subregister modeling: Uses QACC_H_HIGH (128-bit subregister) to enable parallel scheduling.
// CRITICAL: Uses = []<Register> to eliminate implicit Uses and prevent "undefined physical register" error.
// The explicit QACC_H_HIGH:$src virtual register input is sufficient for register allocation.
let Uses = []<Register> in
def ESP_ST_QACC_H_H_128_IP: EspVInst<(outs GPRPIE:$rs1r), (ins QACC_H_HIGH:$src, GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.h.h.128.ip\t $rs1, $off25616", 
                   [(set GPRPIE:$rs1r, (riscv_esp_st_qacc_h_h_128_ip_m QACC_H_HIGH:$src, GPRPIE:$rs1, offset_256_16:$off25616))]> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_ST_QACC_H_H_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_h_h_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_h_h_128_ip GPRPIE:$rs1, timm:$off25616))]>;

// ESP_ST_QACC_H_L_128_IP: Store QACC_H[127:0] to memory
// CRITICAL: Uses = []<Register> to eliminate implicit Uses and prevent "undefined physical register" error.
// The explicit QACC_H_LOW:$src virtual register input is sufficient for register allocation.
let Uses = []<Register> in
def ESP_ST_QACC_H_L_128_IP: EspVInst<(outs GPRPIE:$rs1r), (ins QACC_H_LOW:$src, GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.h.l.128.ip\t $rs1, $off25616", 
                   [(set GPRPIE:$rs1r, (riscv_esp_st_qacc_h_l_128_ip_m QACC_H_LOW:$src, GPRPIE:$rs1, offset_256_16:$off25616))]> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_ST_QACC_H_L_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_h_l_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_h_l_128_ip GPRPIE:$rs1, timm:$off25616))]>;

// ESP_ST_QACC_L_H_128_IP: Store QACC_L[255:128] to memory
// Subregister modeling: Uses QACC_L_HIGH (128-bit subregister) to enable parallel scheduling.
// CRITICAL: Uses = []<Register> to eliminate implicit Uses and prevent "undefined physical register" error.
// The explicit QACC_L_HIGH:$src virtual register input is sufficient for register allocation.
let Uses = []<Register> in
def ESP_ST_QACC_L_H_128_IP: EspVInst<(outs GPRPIE:$rs1r), (ins QACC_L_HIGH:$src, GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.l.h.128.ip\t $rs1, $off25616", 
                   [(set GPRPIE:$rs1r, (riscv_esp_st_qacc_l_h_128_ip_m QACC_L_HIGH:$src, GPRPIE:$rs1, offset_256_16:$off25616))]> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_ST_QACC_L_H_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_l_h_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_l_h_128_ip GPRPIE:$rs1, timm:$off25616))]>;

// ESP_ST_QACC_L_L_128_IP: Store QACC_L[127:0] to memory
// Subregister modeling: Uses QACC_L_LOW (128-bit subregister) to enable parallel scheduling.
// CRITICAL: Uses = []<Register> to eliminate implicit Uses and prevent "undefined physical register" error.
// The explicit QACC_L_LOW:$src virtual register input is sufficient for register allocation.
let Uses = []<Register> in
def ESP_ST_QACC_L_L_128_IP: EspVInst<(outs GPRPIE:$rs1r), (ins QACC_L_LOW:$src, GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.l.l.128.ip\t $rs1, $off25616", 
                   [(set GPRPIE:$rs1r, (riscv_esp_st_qacc_l_l_128_ip_m QACC_L_LOW:$src, GPRPIE:$rs1, offset_256_16:$off25616))]> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_ST_QACC_L_L_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_l_l_128_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_l_l_128_ip GPRPIE:$rs1, timm:$off25616))]>;

// ESP.ST.UA.STATE.IP - Store UA_STATE register to memory
// Real instruction with phantom operand (ghost operand for data flow analysis)
// UA_STATEReg:$src is a phantom operand: appears in (ins) for data flow dependency,
// but is NOT printed in the assembly string (only $rs1 and $off25616 are printed)
// This prevents esp.zero.ua_state from being optimized away while keeping assembly clean
// Note: No Uses = [UA_STATE] because we use explicit phantom operands for explicit state passing
def ESP_ST_UA_STATE_IP: EspVInst<(outs UA_STATEReg:$ua_state_out, GPRPIE:$rs1r), (ins UA_STATEReg:$ua_state_in, GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.ua.state.ip\t $rs1, $off25616", 
                   []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // UA_STATE tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $ua_state_out = $ua_state_in";
  
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
def ESP_ST_UA_STATE_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_ua_state_ip_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_ua_state_ip GPRPIE:$rs1, timm:$off25616))]>;

def ESP_STXQ_32: EspVInst<(outs), (ins GPRPIE:$rs1, QR:$qw, QR:$qu, select_4:$sel4, select_8:$sel8),
                   "esp.stxq.32\t $qu, $qw, $rs1, $sel4, $sel8", 
                   []> 
{
  bits<5> rs1;
  bits<3> qw;
  bits<3> qu;
  bits<2> sel4;
  bits<3> sel8;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_STXQ_32_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qw, imm8:$qu, select_4:$sel4, select_8:$sel8),
				"!esp_stxq_32_p $qu, $qw, $rs1, $sel4, $sel8",
			[(int_riscv_esp_stxq_32 GPRPIE:$rs1, timm:$qw, timm:$qu, timm:$sel4, timm:$sel8)]>;

// ESP_VLD_128_IP: Load 128-bit vector with QR_64 outputs
// Outputs QR_64 directly, ensuring they are paired to the same parent QR register
// via QRTuples. This prevents the need for subsequent extraction operations.
// QR_64 contains both low and high 64-bit subregisters (16 physical registers total).
// Similar to how QACC instructions output QACC_L and QACC_H.
def ESP_VLD_128_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.vld.128.ip\t $qu, $rs1r, $off25616", 
                   [(set QR:$qu, GPRPIE:$rs1r, (riscv_esp_vld_128_ip_m GPRPIE:$rs1, offset_256_16:$off25616))]>
{
  bits<5> rs1;
  bits<8> off25616;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLD_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616, imm8:$qu),
				"!esp_vld_128_ip_p $qu, $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_128_ip GPRPIE:$rs1, timm:$off25616, timm:$qu))]>;

// Pattern to match riscv_esp_vld_128_xp_m with register offset
def ESP_VLD_128_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vld.128.xp\t $qu, $rs1, $rs2", 
                   [(set QR:$qu, GPRPIE:$rs1r, (riscv_esp_vld_128_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLD_128_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vld_128_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_128_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLD_H_64_IP: EspVInst<(outs QR_64:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.vld.h.64.ip\t $qu, $rs1, $off2568", 
                   [(set QR_64:$qu, GPRPIE:$rs1r, (riscv_esp_vld_h_64_ip_m GPRPIE:$rs1, offset_256_8:$off2568))]> 
{
  bits<5> rs1;
  bits<8> off2568;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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

let isAsmParserOnly = 1, hasSideEffects = 0 in {

  // 1. esp.vld.h.64.ip
  def ESP_VLD_H_64_IP_ASM : EspVInst<
      (outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
      "esp.vld.h.64.ip\t $qu, $rs1, $off2568", [], NoItinerary> {
    bits<5> rs1;
    bits<8> off2568;
    bits<3> qu;
    bits<5> rs1r;
    let mayLoad = 1; let mayStore = 0;
    let Constraints = "$rs1r = $rs1";

    // copy from ESP_VLD_H_64_IP
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

  // 2. esp.vld.h.64.xp
  def ESP_VLD_H_64_XP_ASM : EspVInst<
      (outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
      "esp.vld.h.64.xp\t $qu, $rs1, $rs2", [], NoItinerary> {
    bits<5> rs2;
    bits<5> rs1;
    bits<3> qu;
    bits<5> rs1r;
    let mayLoad = 1; let mayStore = 0;
    let Constraints = "$rs1r = $rs1";

    // copy from ESP_VLD_H_64_XP
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

  // 3. esp.vld.l.64.ip
  def ESP_VLD_L_64_IP_ASM : EspVInst<
      (outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
      "esp.vld.l.64.ip\t $qu, $rs1, $off2568", [], NoItinerary> {
    bits<5> rs1;
    bits<8> off2568;
    bits<3> qu;
    bits<5> rs1r;
    let mayLoad = 1; let mayStore = 0;
    let Constraints = "$rs1r = $rs1";

    // copy from ESP_VLD_L_64_IP
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

  // 4. esp.vld.l.64.xp
  def ESP_VLD_L_64_XP_ASM : EspVInst<
      (outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
      "esp.vld.l.64.xp\t $qu, $rs1, $rs2", [], NoItinerary> {
    bits<5> rs2;
    bits<5> rs1;
    bits<3> qu;
    bits<5> rs1r;
    let mayLoad = 1; let mayStore = 0;
    let Constraints = "$rs1r = $rs1";

    // copy from ESP_VLD_L_64_XP
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

  // 5. esp.vst.h.64.ip
  def ESP_VST_H_64_IP_ASM : EspVInst<
      (outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
      "esp.vst.h.64.ip\t $qu, $rs1, $off2568", [], NoItinerary> {
    bits<3> qu;
    bits<5> rs1;
    bits<8> off2568;
    bits<5> rs1r;
    let mayLoad = 0; let mayStore = 1;
    let Constraints = "$rs1r = $rs1";

    // copy from ESP_VST_H_64_IP
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

  // 6. esp.vst.h.64.xp
  def ESP_VST_H_64_XP_ASM : EspVInst<
      (outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qu, GPRPIE:$rs1),
      "esp.vst.h.64.xp\t $qu, $rs1, $rs2", [], NoItinerary> {
    bits<5> rs2;
    bits<3> qu;
    bits<5> rs1;
    bits<5> rs1r;
    let mayLoad = 0; let mayStore = 1;
    let Constraints = "$rs1r = $rs1";

    // copy from ESP_VST_H_64_XP
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

  // 7. esp.vst.l.64.ip
  def ESP_VST_L_64_IP_ASM : EspVInst<
      (outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
      "esp.vst.l.64.ip\t $qu, $rs1, $off2568", [], NoItinerary> {
    bits<3> qu;
    bits<5> rs1;
    bits<8> off2568;
    bits<5> rs1r;
    let mayLoad = 0; let mayStore = 1;
    let Constraints = "$rs1r = $rs1";

    // copy from ESP_VST_L_64_IP
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

  // 8. esp.vst.l.64.xp
  def ESP_VST_L_64_XP_ASM : EspVInst<
      (outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qu, GPRPIE:$rs1),
      "esp.vst.l.64.xp\t $qu, $rs1, $rs2", [], NoItinerary> {
    bits<5> rs2;
    bits<3> qu;
    bits<5> rs1;
    bits<5> rs1r;
    let mayLoad = 0; let mayStore = 1;
    let Constraints = "$rs1r = $rs1";

    // copy from ESP_VST_L_64_XP
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
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_H_64_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568, imm8:$qu),
				"!esp_vld_h_64_ip_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_h_64_ip GPRPIE:$rs1, timm:$off2568, timm:$qu))]>;

def ESP_VLD_H_64_XP: EspVInst<(outs QR_64:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vld.h.64.xp\t $qu, $rs1, $rs2", 
                   [(set QR_64:$qu, GPRPIE:$rs1r, (riscv_esp_vld_h_64_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLD_H_64_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vld_h_64_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_h_64_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLD_L_64_IP: EspVInst<(outs QR_64:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
"esp.vld.l.64.ip\t $qu, $rs1, $off2568", 
                   [(set QR_64:$qu, GPRPIE:$rs1r, (riscv_esp_vld_l_64_ip_m GPRPIE:$rs1, offset_256_8:$off2568))]>
{
  bits<5> rs1;
  bits<8> off2568;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLD_L_64_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568, imm8:$qu),
				"!esp_vld_l_64_ip_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_l_64_ip GPRPIE:$rs1, timm:$off2568, timm:$qu))]>;

def ESP_VLD_L_64_XP: EspVInst<(outs QR_64:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.vld.l.64.xp\t $qu, $rs1, $rs2", 
                   [(set QR_64:$qu, GPRPIE:$rs1r, (riscv_esp_vld_l_64_xp_m GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  
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
def ESP_VLD_L_64_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vld_l_64_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_l_64_xp GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VST_128_IP: EspVInst<(outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.vst.128.ip\t $qu, $rs1r, $off25616", 
                   [(set GPRPIE:$rs1r, (riscv_esp_vst_128_ip_m QR:$qu, GPRPIE:$rs1, offset_256_16:$off25616))]> 
{
  bits<3> qu;
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VST_128_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_vst_128_ip_p $qu, $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_128_ip timm:$qu, GPRPIE:$rs1, timm:$off25616))]>;

def ESP_VST_128_XP: EspVInst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qu, GPRPIE:$rs1),
                   "esp.vst.128.xp\t $qu, $rs1, $rs2", 
                   [(set GPRPIE:$rs1r, (riscv_esp_vst_128_xp_m QR:$qu, GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VST_128_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qu, GPRPIE:$rs1),
				"!esp_vst_128_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_128_xp GPRPIE:$rs2, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VST_H_64_IP: EspVInst<(outs GPRPIE:$rs1r), (ins QR_64:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.vst.h.64.ip\t $qu, $rs1, $off2568", 
                   [(set GPRPIE:$rs1r, (riscv_esp_vst_h_64_ip_m QR_64:$qu, GPRPIE:$rs1, offset_256_8:$off2568))]> 
{
  bits<3> qu;
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VST_H_64_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_vst_h_64_ip_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_h_64_ip timm:$qu, GPRPIE:$rs1, timm:$off2568))]>;

def ESP_VST_H_64_XP: EspVInst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR_64:$qu, GPRPIE:$rs1),
                   "esp.vst.h.64.xp\t $qu, $rs1, $rs2", 
                   [(set GPRPIE:$rs1r, (riscv_esp_vst_h_64_xp_m QR_64:$qu, GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VST_H_64_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qu, GPRPIE:$rs1),
				"!esp_vst_h_64_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_h_64_xp GPRPIE:$rs2, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VST_L_64_IP: EspVInst<(outs GPRPIE:$rs1r), (ins QR_64:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.vst.l.64.ip\t $qu, $rs1, $off2568", 
                   [(set GPRPIE:$rs1r, (riscv_esp_vst_l_64_ip_m QR_64:$qu, GPRPIE:$rs1, offset_256_8:$off2568))]> 
{
  bits<3> qu;
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VST_L_64_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_vst_l_64_ip_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_l_64_ip timm:$qu, GPRPIE:$rs1, timm:$off2568))]>;

def ESP_VST_L_64_XP: EspVInst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR_64:$qu, GPRPIE:$rs1),
                   "esp.vst.l.64.xp\t $qu, $rs1, $rs2", 
                   [(set GPRPIE:$rs1r, (riscv_esp_vst_l_64_xp_m QR_64:$qu, GPRPIE:$rs1, GPRPIE:$rs2))]> 
{
  bits<5> rs2;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VST_L_64_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qu, GPRPIE:$rs1),
				"!esp_vst_l_64_xp_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_l_64_xp GPRPIE:$rs2, timm:$qu, GPRPIE:$rs1))]>;

def ESP_SLCI_2Q: EspVInst<(outs QR:$qyr, QR:$qwr), (ins QR:$qy, QR:$qw, select_16:$sel16),
                   "esp.slci.2q\t $qy, $qw, $sel16", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<4> sel16;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SLCI_2Q_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, select_16:$sel16),
				"!esp_slci_2q_p $qy, $qw, $sel16",
			[(int_riscv_esp_slci_2q timm:$qy, timm:$qw, timm:$sel16)]>;

def ESP_SLCXXP_2Q: EspVInst<(outs QR:$qyr, QR:$qwr, GPRPIE:$rs1r), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy, QR:$qw),
                   "esp.slcxxp.2q\t $qy, $qw, $rs1, $rs2", 
                   []>
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qy;
  bits<3> qw;
  bits<3> qyr;
  bits<3> qwr;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;  // Explicit state passing: all outputs visible in IR
  
  // Tied constraints: input and output registers use same physical register
  // This models hardware's in-place update behavior
  let Constraints = "$qyr = $qy, $qwr = $qw, $rs1r = $rs1";
  
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
def ESP_SLCXXP_2Q_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy, imm8:$qw),
				"!esp_slcxxp_2q_p $qy, $qw, $rs1, $rs2",
			[(int_riscv_esp_slcxxp_2q GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy, timm:$qw)]>;

// ESP.SRC.Q with explicit SAR_BYTES input
// SAR_BYTES is explicit input for explicit state passing
def ESP_SRC_Q: EspVInst<(outs QR:$qz), (ins SAR_BYTESReg:$sar_bytes_in, QR:$qy, QR:$qw),
                   "esp.src.q\t $qz, $qw, $qy", 
                   // Note: Shift amount is SAR_BYTES[3:0] << 3 (hardware automatically uses only low 4 bits)
                   []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRC_Q_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qz),
				"!esp_src_q_p $qz, $qw, $qy",
			[(int_riscv_esp_src_q timm:$qy, timm:$qw, timm:$qz)]>;

// ESP.SRC.Q.LD.IP with explicit SAR_BYTES input
// SAR_BYTES is explicit input for explicit state passing
def ESP_SRC_Q_LD_IP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QR:$qwr), (ins SAR_BYTESReg:$sar_bytes_in, QR:$qy, GPRPIE:$rs1, QR:$qw, offset_256_16:$off25616),
                   "esp.src.q.ld.ip\t $qu, $rs1, $off25616, $qw, $qy", 
                   // Note: Shift amount is SAR_BYTES[3:0] << 3 (hardware automatically uses only low 4 bits)
                   []>
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
  let hasSideEffects = 0;
  
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
def ESP_SRC_Q_LD_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qy, GPRPIE:$rs1, imm8:$qw, offset_256_16:$off25616, imm8:$qu),
				"!esp_src_q_ld_ip_p $qu, $rs1, $off25616, $qw, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_src_q_ld_ip timm:$qy, GPRPIE:$rs1, timm:$qw, timm:$off25616, timm:$qu))]>;

// ESP.SRC.Q.LD.XP with explicit SAR_BYTES input
// SAR_BYTES is explicit input for explicit state passing
// SDNode operand order: SAR_BYTES, qy, qw, ptr, offset_reg
def ESP_SRC_Q_LD_XP: EspVInst<(outs QR:$qu, GPRPIE:$rs1r, QR:$qwr), (ins SAR_BYTESReg:$sar_bytes_in, QR:$qy, QR:$qw, GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.src.q.ld.xp\t $qu, $rs1, $rs2, $qw, $qy", 
                   // Note: Shift amount is SAR_BYTES[3:0] << 3 (hardware automatically uses only low 4 bits)
                   // SDNode operand order matches SDTypeProfile: SAR_BYTES, qy, qw, ptr, offset_reg
                   []> 
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
  let hasSideEffects = 0;
  
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
def ESP_SRC_Q_LD_XP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qy, GPRPIE:$rs1, imm8:$qw, imm8:$qu),
				"!esp_src_q_ld_xp_p $qu, $rs1, $rs2, $qw, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_src_q_ld_xp GPRPIE:$rs2, timm:$qy, GPRPIE:$rs1, timm:$qw, timm:$qu))]>;

// ESP.SRC.Q.qup with explicit SAR_BYTES input
// SAR_BYTES is explicit input for explicit state passing
def ESP_SRC_Q_QUP: EspVInst<(outs QR:$qz, QR:$qwr), (ins SAR_BYTESReg:$sar_bytes_in, QR:$qy, QR:$qw),
                   "esp.src.q.qup\t $qz, $qw, $qy", 
                   // Note: Shift amount is SAR_BYTES[3:0] << 3 (hardware automatically uses only low 4 bits)
                   []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qz;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRC_Q_QUP_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qz),
				"!esp_src_q_qup_p $qz, $qw, $qy",
			[(int_riscv_esp_src_q_qup timm:$qy, timm:$qw, timm:$qz)]>;

def ESP_SRCI_2Q: EspVInst<(outs QR:$qyr, QR:$qwr), (ins QR:$qy, QR:$qw, select_16:$sel16),
                   "esp.srci.2q\t $qy, $qw, $sel16", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<4> sel16;
  bits<3> qyr;
  bits<3> qwr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRCI_2Q_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, select_16:$sel16),
				"!esp_srci_2q_p $qy, $qw, $sel16",
			[(int_riscv_esp_srci_2q timm:$qy, timm:$qw, timm:$sel16)]>;

// SRCI.2Q operation (_m version)
// Shifts concatenated qy and qw right, updates both registers in-place
// Input and output registers are the same due to constraints
// Pattern is empty - custom inserter handles everything including register mapping

def ESP_SRCMB_S16_Q_QACC: EspVInst<(outs QR:$qu), (ins QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, QR:$qw, select_2:$sel2),
                   "esp.srcmb.s16.q.qacc\t $qu, $qw, $sel2", 
                   []> 
{
  bits<3> qw;
  bits<1> sel2;
  bits<3> qu;
  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRCMB_S16_Q_QACC_P : PseudoEsp<(outs), (ins imm8:$qw, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_s16_q_qacc_p $qu, $qw, $sel2",
			[(int_riscv_esp_srcmb_s16_q_qacc timm:$qw, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_S16_QACC: EspVInst<(outs QR:$qu), (ins QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1, select_2:$sel2),
                   "esp.srcmb.s16.qacc\t $qu, $rs1, $sel2",
                   [(set QR:$qu, (riscv_esp_srcmb_s16_qacc_m QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1, select_2:$sel2))]>
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
def ESP_SRCMB_S16_QACC_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_s16_qacc_p $qu, $rs1, $sel2",
			[(int_riscv_esp_srcmb_s16_qacc GPRPIE:$rs1, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_S8_Q_QACC: EspVInst<(outs QR:$qu), (ins QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, QR:$qw, select_2:$sel2),
                   "esp.srcmb.s8.q.qacc\t $qu, $qw, $sel2", 
                   []> 
{
  bits<3> qw;
  bits<1> sel2;
  bits<3> qu;
  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRCMB_S8_Q_QACC_P : PseudoEsp<(outs), (ins imm8:$qw, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_s8_q_qacc_p $qu, $qw, $sel2",
			[(int_riscv_esp_srcmb_s8_q_qacc timm:$qw, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_S8_QACC: EspVInst<(outs QR:$qu), (ins QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1, select_2:$sel2),
                   "esp.srcmb.s8.qacc\t $qu, $rs1, $sel2", 
                   []> 
{
  bits<5> rs1;
  bits<1> sel2;
  bits<3> qu;
  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRCMB_S8_QACC_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_s8_qacc_p $qu, $rs1, $sel2",
			[(int_riscv_esp_srcmb_s8_qacc GPRPIE:$rs1, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_U16_Q_QACC: EspVInst<(outs QR:$qu), (ins QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, QR:$qw, select_2:$sel2),
                   "esp.srcmb.u16.q.qacc\t $qu, $qw, $sel2", 
                   []> 
{
  bits<3> qw;
  bits<1> sel2;
  bits<3> qu;
  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRCMB_U16_Q_QACC_P : PseudoEsp<(outs), (ins imm8:$qw, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_u16_q_qacc_p $qu, $qw, $sel2",
			[(int_riscv_esp_srcmb_u16_q_qacc timm:$qw, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_U16_QACC: EspVInst<(outs QR:$qu), (ins QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1, select_2:$sel2),
                   "esp.srcmb.u16.qacc\t $qu, $rs1, $sel2",
                   []>
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
def ESP_SRCMB_U16_QACC_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_u16_qacc_p $qu, $rs1, $sel2",
			[(int_riscv_esp_srcmb_u16_qacc GPRPIE:$rs1, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_U8_Q_QACC: EspVInst<(outs QR:$qu), (ins QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, QR:$qw, select_2:$sel2),
                   "esp.srcmb.u8.q.qacc\t $qu, $qw, $sel2", 
                   []> 
{
  bits<3> qw;
  bits<1> sel2;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRCMB_U8_Q_QACC_P : PseudoEsp<(outs), (ins imm8:$qw, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_u8_q_qacc_p $qu, $qw, $sel2",
			[(int_riscv_esp_srcmb_u8_q_qacc timm:$qw, timm:$sel2, timm:$qu)]>;

def ESP_SRCMB_U8_QACC: EspVInst<(outs QR:$qu), (ins QACC_L_LOW:$qacc_l_l, QACC_L_HIGH:$qacc_l_h, QACC_H_LOW:$qacc_h_l, QACC_H_HIGH:$qacc_h_h, GPRPIE:$rs1, select_2:$sel2),
                   "esp.srcmb.u8.qacc\t $qu, $rs1, $sel2",
                   []>
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
def ESP_SRCMB_U8_QACC_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, select_2:$sel2, imm8:$qu),
				"!esp_srcmb_u8_qacc_p $qu, $rs1, $sel2",
			[(int_riscv_esp_srcmb_u8_qacc GPRPIE:$rs1, timm:$sel2, timm:$qu)]>;

// ESP.SRCQ.128.ST.INCP with explicit SAR_BYTES input
// SAR_BYTES is explicit input for explicit state passing
def ESP_SRCQ_128_ST_INCP: EspVInst<(outs GPRPIE:$rs1r), (ins SAR_BYTESReg:$sar_bytes_in, QR:$qy, QR:$qw, GPRPIE:$rs1),
                   "esp.srcq.128.st.incp\t $qw, $qy, $rs1", 
                   // Note: Shift amount is SAR_BYTES[3:0] << 3 (hardware automatically uses only low 4 bits)
                   []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRCQ_128_ST_INCP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qy, imm8:$qw, GPRPIE:$rs1),
				"!esp_srcq_128_st_incp_p $qw, $qy, $rs1",
			[(set GPRPIE:$rs1r, (int_riscv_esp_srcq_128_st_incp timm:$qy, timm:$qw, GPRPIE:$rs1))]>;

def ESP_SRCXXP_2Q: EspVInst<(outs QR:$qyr, QR:$qwr, GPRPIE:$rs1r), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy, QR:$qw),
                   "esp.srcxxp.2q\t $qy, $qw, $rs1, $rs2", 
                   []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qy;
  bits<3> qw;
  bits<3> qyr;
  bits<3> qwr;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;  // Explicit state passing: all outputs visible in IR
  
  // Tied constraints: input and output registers use same physical register
  // This models hardware's in-place update behavior
  let Constraints = "$qyr = $qy, $qwr = $qw, $rs1r = $rs1";
  
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
def ESP_SRCXXP_2Q_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy, imm8:$qw),
				"!esp_srcxxp_2q_p $qy, $qw, $rs1, $rs2",
			[(int_riscv_esp_srcxxp_2q GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy, timm:$qw)]>;

// ESP.SRS.S.XACC (_m version) - Shift Right and Saturate Signed with explicit state passing
// XACC is explicit input and output operand for chaining
// Intrinsic returns {i32 saturated_value, i64 new_xacc}
def ESP_SRS_S_XACC: EspVInst<(outs GPRPIE:$rd, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs1),
                   "esp.srs.s.xacc\t $rd, $rs1", 
                   // Mixed model: XACC as {i32 low, i32 high}
                   // XACC is explicit in intrinsic (takes XACC as {i32 low, i32 high} input, returns new XACC as {i32 low, i32 high})
                   // Instruction performs arithmetic right shift on XACC, saturates to 32-bit signed, writes back to XACC
                   // Return value: saturated 32-bit signed value + new XACC {i32 low, i32 high}
                   // Pattern matches SDNode which returns {i32, i32, i32} to instruction outputs {GPRPIE:$rd, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out}
                   []> 
{
  bits<5> rs1;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRS_S_XACC_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rd),
				"!esp_srs_s_xacc_p $rd, $rs1",
			[(int_riscv_esp_srs_s_xacc GPRPIE:$rs1, GPRPIE:$rd)]>;

// ESP.SRS.U.XACC (_m version) - Shift Right and Saturate Unsigned with explicit state passing
// XACC is explicit input and output operand for chaining
// Intrinsic returns {i32 saturated_value, i64 new_xacc}
def ESP_SRS_U_XACC: EspVInst<(outs GPRPIE:$rd, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs1),
                   "esp.srs.u.xacc\t $rd, $rs1", 
                   // Mixed model: XACC as {i32 low, i32 high}
                   // XACC is explicit in intrinsic (takes XACC as {i32 low, i32 high} input, returns new XACC as {i32 low, i32 high})
                   // Instruction performs arithmetic right shift on XACC, saturates to 32-bit unsigned, writes back to XACC
                   // Return value: saturated 32-bit unsigned value + new XACC {i32 low, i32 high}
                   // Pattern matches SDNode which returns {i32, i32, i32} to instruction outputs {GPRPIE:$rd, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out}
                   []>
{
  bits<5> rs1;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_SRS_U_XACC_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rd),
				"!esp_srs_u_xacc_p $rd, $rs1",
			[(int_riscv_esp_srs_u_xacc GPRPIE:$rs1, GPRPIE:$rd)]>;

// ESP.VSL.32 reads from SAR register (SAR is input source)
// For explicit state passing, SAR (64-bit) is passed as explicit parameter in intrinsic
// Instruction: qu[127:0] = (qy[127:0] << SAR[5:0]) (hardware extracts only low 6 bits from SAR)
def ESP_VSL_32: EspVInst<(outs QR:$qu), (ins QR:$qy, SARReg:$sar_in),
                   "esp.vsl.32\t $qu, $qy", 
                   // SAR is explicit in intrinsic (SAR is input parameter, i32)
                   // Hardware automatically extracts SAR[5:0] for shift amount
                   [(set QR:$qu, (int_riscv_esp_vsl_32_m QR:$qy, SARReg:$sar_in))]> 
{
  bits<3> qy;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSL_32_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qu),
				"!esp_vsl_32_p $qu, $qy",
			[(int_riscv_esp_vsl_32 timm:$qy, timm:$qu)]>;

def ESP_VSLD_16: EspVInst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsld.16\t $qu, $qy, $qw", 
                   [(set QR:$qu, (int_riscv_esp_vsld_16_m QR:$qy, QR:$qw))]> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSLD_16_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsld_16_p $qu, $qy, $qw",
			[(int_riscv_esp_vsld_16 timm:$qy, timm:$qw, timm:$qu)]>;

def ESP_VSLD_32: EspVInst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsld.32\t $qu, $qy, $qw", 
                   []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSLD_32_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsld_32_p $qu, $qy, $qw",
			[(int_riscv_esp_vsld_32 timm:$qy, timm:$qw, timm:$qu)]>;

def ESP_VSLD_8: EspVInst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsld.8\t $qu, $qy, $qw", 
                   [(set QR:$qu, (int_riscv_esp_vsld_8_m QR:$qy, QR:$qw))]> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSLD_8_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsld_8_p $qu, $qy, $qw",
			[(int_riscv_esp_vsld_8 timm:$qy, timm:$qw, timm:$qu)]>;

// ESP.VSR.S32 reads from SAR register (SAR is input source)
// For explicit state passing, SAR (64-bit) is passed as explicit parameter in intrinsic
// Instruction: qu[127:0] = (qy[127:0] >> SAR[5:0]) (hardware extracts only low 6 bits from SAR)
def ESP_VSR_S32: EspVInst<(outs QR:$qu), (ins QR:$qy, SARReg:$sar_in),
                   "esp.vsr.s32\t $qu, $qy", 
                   // SAR is explicit in intrinsic (SAR is input parameter, i32)
                   // Hardware automatically extracts SAR[5:0] for shift amount
                   [(set QR:$qu, (int_riscv_esp_vsr_s32_m QR:$qy, SARReg:$sar_in))]> 
{
  bits<3> qy;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSR_S32_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qu),
				"!esp_vsr_s32_p $qu, $qy",
			[(int_riscv_esp_vsr_s32 timm:$qy, timm:$qu)]>;

// ESP.VSR.U32 reads from SAR register (SAR is input source)
// For explicit state passing, SAR (64-bit) is passed as explicit parameter in intrinsic
// Instruction: qu[127:0] = (qy[127:0] >> SAR[5:0]) (hardware extracts only low 6 bits from SAR)
def ESP_VSR_U32: EspVInst<(outs QR:$qu), (ins QR:$qy, SARReg:$sar_in),
                   "esp.vsr.u32\t $qu, $qy", 
                   // SAR is explicit in intrinsic (SAR is input parameter, i32)
                   // Hardware automatically extracts SAR[5:0] for shift amount
                   [(set QR:$qu, (int_riscv_esp_vsr_u32_m QR:$qy, SARReg:$sar_in))]> 
{
  bits<3> qy;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSR_U32_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qu),
				"!esp_vsr_u32_p $qu, $qy",
			[(int_riscv_esp_vsr_u32 timm:$qy, timm:$qu)]>;

def ESP_VSRD_16: EspVInst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsrd.16\t $qu, $qy, $qw", 
                   []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSRD_16_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsrd_16_p $qu, $qy, $qw",
			[(int_riscv_esp_vsrd_16 timm:$qy, timm:$qw, timm:$qu)]>;

def ESP_VSRD_32: EspVInst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsrd.32\t $qu, $qy, $qw", 
                   []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSRD_32_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsrd_32_p $qu, $qy, $qw",
			[(int_riscv_esp_vsrd_32 timm:$qy, timm:$qw, timm:$qu)]>;

def ESP_VSRD_8: EspVInst<(outs QR:$qu), (ins QR:$qy, QR:$qw),
                   "esp.vsrd.8\t $qu, $qy, $qw", 
                   [(set QR:$qu, (int_riscv_esp_vsrd_8_m QR:$qy, QR:$qw))]> 
{
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
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
def ESP_VSRD_8_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qu),
				"!esp_vsrd_8_p $qu, $qy, $qw",
			[(int_riscv_esp_vsrd_8 timm:$qy, timm:$qw, timm:$qu)]>;

// Real instruction with phantom operand (ghost operand for data flow analysis)
// XACCReg:$src is a phantom operand: appears in (ins) for data flow dependency,
// but is NOT printed in the assembly string (only $rs1 and $off2568 are printed)
// Mixed model: XACC as {i32 low, i32 high}
// Hardware instruction operates on implicit XACC register, but inputs are explicit for data flow
// Pattern matching: SDNode takes {i32, i32} inputs, instruction takes XACC_LOW and XACC_HIGH as phantom operands
// This prevents esp.zero.xacc from being optimized away while keeping assembly clean
def ESP_ST_S_XACC_IP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.st.s.xacc.ip\t $rs1, $off2568", 
                   []> 
{
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_ST_S_XACC_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_st_s_xacc_ip_p $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_s_xacc_ip GPRPIE:$rs1, timm:$off2568))]>;


// Real instruction with phantom operand (ghost operand for data flow analysis)
// XACCReg:$src is a phantom operand: appears in (ins) for data flow dependency,
// but is NOT printed in the assembly string (only $rs1 and $off2568 are printed)
// Mixed model: XACC as {i32 low, i32 high}
// Hardware instruction operates on implicit XACC register, but inputs are explicit for data flow
// Pattern matching: SDNode takes {i32, i32} inputs, instruction takes XACC_LOW and XACC_HIGH as phantom operands
// This prevents esp.zero.xacc from being optimized away while keeping assembly clean
def ESP_ST_U_XACC_IP: EspVInst<(outs GPRPIE:$rs1r, XACC_LOW:$xacc_low_out, XACC_HIGH:$xacc_high_out), (ins XACC_LOW:$xacc_low_in, XACC_HIGH:$xacc_high_in, GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.st.u.xacc.ip\t $rs1, $off2568", 
                   []>
{
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 0;
  
  // Tied constraints: rs1r = rs1 (pointer update)
  // XACC tied constraints: Output equals input (passthru), must use same physical register
  // This is critical for singleton register classes (only one physical register)
  let Constraints = "$rs1r = $rs1, $xacc_low_out = $xacc_low_in, $xacc_high_out = $xacc_high_in";
  
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
def ESP_ST_U_XACC_IP_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_st_u_xacc_ip_p $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_u_xacc_ip GPRPIE:$rs1, timm:$off2568))]>;

include "RISCVInstrInfoESP32P4Patterns.td"

def ESP_VCMULAS_S16_QACC_H_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vcmulas.s16.qacc.h\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMULAS_S16_QACC_H_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vcmulas_s16_qacc_h_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vcmulas_s16_qacc_h_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VCMULAS_S16_QACC_H_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vcmulas.s16.qacc.h.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{23} = sat{0};
  let Inst{22-19} = off1616{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S16_QACC_H_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s16_qacc_h_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_h_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S16_QACC_H_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vcmulas.s16.qacc.h.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S16_QACC_H_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vcmulas_s16_qacc_h_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_h_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VCMULAS_S16_QACC_L_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vcmulas.s16.qacc.l\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMULAS_S16_QACC_L_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vcmulas_s16_qacc_l_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vcmulas_s16_qacc_l_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VCMULAS_S16_QACC_L_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vcmulas.s16.qacc.l.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{23} = sat{0};
  let Inst{22-19} = off1616{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S16_QACC_L_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s16_qacc_l_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_l_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S16_QACC_L_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vcmulas.s16.qacc.l.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S16_QACC_L_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vcmulas_s16_qacc_l_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s16_qacc_l_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VCMULAS_S8_QACC_H_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vcmulas.s8.qacc.h\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMULAS_S8_QACC_H_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vcmulas_s8_qacc_h_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vcmulas_s8_qacc_h_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VCMULAS_S8_QACC_H_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vcmulas.s8.qacc.h.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{23} = sat{0};
  let Inst{22-19} = off1616{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S8_QACC_H_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s8_qacc_h_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_h_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S8_QACC_H_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vcmulas.s8.qacc.h.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S8_QACC_H_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vcmulas_s8_qacc_h_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_h_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VCMULAS_S8_QACC_L_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vcmulas.s8.qacc.l\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMULAS_S8_QACC_L_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vcmulas_s8_qacc_l_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vcmulas_s8_qacc_l_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VCMULAS_S8_QACC_L_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vcmulas.s8.qacc.l.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{23} = sat{0};
  let Inst{22-19} = off1616{3-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S8_QACC_L_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vcmulas_s8_qacc_l_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_l_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VCMULAS_S8_QACC_L_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vcmulas.s8.qacc.l.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VCMULAS_S8_QACC_L_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vcmulas_s8_qacc_l_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vcmulas_s8_qacc_l_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_S16_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vmulas.s16.qacc\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = sat{0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_S16_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vmulas_s16_qacc_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vmulas_s16_qacc_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VMULAS_S16_QACC_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.s16.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_QACC_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s16_qacc_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_S16_QACC_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.s16.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_QACC_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_s16_qacc_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_S16_QACC_ST_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.s16.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_QACC_ST_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
				"!esp_vmulas_s16_qacc_st_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_st_ip_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$off1616))]>;

def ESP_VMULAS_S16_QACC_ST_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.s16.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_QACC_ST_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat),
				"!esp_vmulas_s16_qacc_st_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_st_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat))]>;

def ESP_VMULAS_S16_XACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vmulas.s16.xacc\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 1;
  let Inst{15} = sat{0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_S16_XACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vmulas_s16_xacc_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vmulas_s16_xacc_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VMULAS_S16_XACC_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.s16.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_XACC_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s16_xacc_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_S16_XACC_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.s16.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_XACC_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_s16_xacc_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_S16_XACC_ST_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.s16.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_XACC_ST_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
				"!esp_vmulas_s16_xacc_st_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_st_ip_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$off1616))]>;

def ESP_VMULAS_S16_XACC_ST_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.s16.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_XACC_ST_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat),
				"!esp_vmulas_s16_xacc_st_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_xacc_st_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat))]>;

def ESP_VMULAS_S8_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vmulas.s8.qacc\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = sat{0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_S8_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vmulas_s8_qacc_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vmulas_s8_qacc_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VMULAS_S8_QACC_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.s8.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_QACC_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s8_qacc_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_S8_QACC_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.s8.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_QACC_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_s8_qacc_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_S8_QACC_ST_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.s8.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_QACC_ST_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
				"!esp_vmulas_s8_qacc_st_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_st_ip_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$off1616))]>;

def ESP_VMULAS_S8_QACC_ST_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.s8.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_QACC_ST_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat),
				"!esp_vmulas_s8_qacc_st_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_st_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat))]>;

def ESP_VMULAS_S8_XACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vmulas.s8.xacc\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = sat{0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_S8_XACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vmulas_s8_xacc_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vmulas_s8_xacc_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VMULAS_S8_XACC_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.s8.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_XACC_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_s8_xacc_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_S8_XACC_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.s8.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_XACC_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_s8_xacc_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_S8_XACC_ST_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.s8.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_XACC_ST_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
				"!esp_vmulas_s8_xacc_st_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_st_ip_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$off1616))]>;

def ESP_VMULAS_S8_XACC_ST_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.s8.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_XACC_ST_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat),
				"!esp_vmulas_s8_xacc_st_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_xacc_st_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat))]>;

def ESP_VMULAS_U16_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vmulas.u16.qacc\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = sat{0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_U16_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vmulas_u16_qacc_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vmulas_u16_qacc_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VMULAS_U16_QACC_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.u16.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_QACC_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u16_qacc_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_U16_QACC_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.u16.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_QACC_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_u16_qacc_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_U16_QACC_ST_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.u16.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_QACC_ST_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
				"!esp_vmulas_u16_qacc_st_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_st_ip_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$off1616))]>;

def ESP_VMULAS_U16_QACC_ST_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.u16.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_QACC_ST_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat),
				"!esp_vmulas_u16_qacc_st_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_st_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat))]>;

def ESP_VMULAS_U16_XACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vmulas.u16.xacc\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = sat{0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_U16_XACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vmulas_u16_xacc_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vmulas_u16_xacc_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VMULAS_U16_XACC_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.u16.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_XACC_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u16_xacc_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_U16_XACC_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.u16.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_XACC_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_u16_xacc_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_U16_XACC_ST_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.u16.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_XACC_ST_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
				"!esp_vmulas_u16_xacc_st_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_st_ip_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$off1616))]>;

def ESP_VMULAS_U16_XACC_ST_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.u16.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_XACC_ST_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat),
				"!esp_vmulas_u16_xacc_st_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_xacc_st_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat))]>;

def ESP_VMULAS_U8_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vmulas.u8.qacc\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = sat{0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_U8_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vmulas_u8_qacc_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vmulas_u8_qacc_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VMULAS_U8_QACC_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.u8.qacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_QACC_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u8_qacc_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_U8_QACC_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.u8.qacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_QACC_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_u8_qacc_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_U8_QACC_ST_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.u8.qacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_QACC_ST_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
				"!esp_vmulas_u8_qacc_st_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_st_ip_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$off1616))]>;

def ESP_VMULAS_U8_QACC_ST_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.u8.qacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_QACC_ST_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat),
				"!esp_vmulas_u8_qacc_st_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_st_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat))]>;

def ESP_VMULAS_U8_XACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vmulas.u8.xacc\t $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = sat{0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMULAS_U8_XACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat),
				"!esp_vmulas_u8_xacc_2p2_p $qx, $qy, $sat",
			[(int_riscv_esp_vmulas_u8_xacc_2p2 timm:$qx, timm:$qy, timm:$sat)]>;

def ESP_VMULAS_U8_XACC_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.u8.xacc.ld.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_XACC_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616, imm8:$qu),
				"!esp_vmulas_u8_xacc_ld_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_ld_ip_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$off1616, timm:$qu))]>;

def ESP_VMULAS_U8_XACC_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.u8.xacc.ld.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_XACC_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_u8_xacc_ld_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_U8_XACC_ST_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
                   "esp.vmulas.u8.xacc.st.ip\t $qu, $rs1, $off1616, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20-19} = off1616{3-2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off1616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_XACC_ST_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, offset_16_16:$off1616),
				"!esp_vmulas_u8_xacc_st_ip_2p2_p $qu, $rs1, $off1616, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_st_ip_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$off1616))]>;

def ESP_VMULAS_U8_XACC_ST_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.u8.xacc.st.xp\t $qu, $rs1, $rs2, $qx, $qy, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{8} = sat{0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_XACC_ST_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat),
				"!esp_vmulas_u8_xacc_st_xp_2p2_p $qu, $rs1, $rs2, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_xacc_st_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat))]>;

def ESP_VMULAS_S16_QACC_LDBC_INCP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.s16.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S16_QACC_LDBC_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_s16_qacc_ldbc_incp_2p2_p $qu, $rs1, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s16_qacc_ldbc_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_S8_QACC_LDBC_INCP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.s8.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_S8_QACC_LDBC_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_s8_qacc_ldbc_incp_2p2_p $qu, $rs1, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_s8_qacc_ldbc_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_U16_QACC_LDBC_INCP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.u16.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U16_QACC_LDBC_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_u16_qacc_ldbc_incp_2p2_p $qu, $rs1, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u16_qacc_ldbc_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VMULAS_U8_QACC_LDBC_INCP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vmulas.u8.qacc.ldbc.incp\t $qu, $rs1, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{21} = sat{0};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VMULAS_U8_QACC_LDBC_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qu),
				"!esp_vmulas_u8_qacc_ldbc_incp_2p2_p $qu, $rs1, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmulas_u8_qacc_ldbc_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qu))]>;

def ESP_VSMULAS_S16_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat, select_16:$sel16),
                   "esp.vsmulas.s16.qacc\t $qx, $qy, $sel16, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{20} = sat{0};
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSMULAS_S16_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, select_16:$sel16),
				"!esp_vsmulas_s16_qacc_2p2_p $qx, $qy, $sel16, $sat",
			[(int_riscv_esp_vsmulas_s16_qacc_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$sel16)]>;

def ESP_VSMULAS_S16_QACC_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, select_16:$sel16),
                   "esp.vsmulas.s16.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{23} = sat{0};
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
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSMULAS_S16_QACC_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_s16_qacc_ld_incp_2p2_p $qu, $rs1, $qx, $qy, $sel16, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_s16_qacc_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$sel16, timm:$qu))]>;

def ESP_VSMULAS_S8_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat, select_16:$sel16),
                   "esp.vsmulas.s8.qacc\t $qx, $qy, $sel16, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{20} = sat{0};
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSMULAS_S8_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, select_16:$sel16),
				"!esp_vsmulas_s8_qacc_2p2_p $qx, $qy, $sel16, $sat",
			[(int_riscv_esp_vsmulas_s8_qacc_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$sel16)]>;

def ESP_VSMULAS_S8_QACC_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, select_16:$sel16),
                   "esp.vsmulas.s8.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{23} = sat{0};
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
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSMULAS_S8_QACC_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_s8_qacc_ld_incp_2p2_p $qu, $rs1, $qx, $qy, $sel16, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_s8_qacc_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$sel16, timm:$qu))]>;

def ESP_VSMULAS_U16_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat, select_16:$sel16),
                   "esp.vsmulas.u16.qacc\t $qx, $qy, $sel16, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{20} = sat{0};
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSMULAS_U16_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, select_16:$sel16),
				"!esp_vsmulas_u16_qacc_2p2_p $qx, $qy, $sel16, $sat",
			[(int_riscv_esp_vsmulas_u16_qacc_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$sel16)]>;

def ESP_VSMULAS_U16_QACC_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, select_16:$sel16),
                   "esp.vsmulas.u16.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{23} = sat{0};
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
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSMULAS_U16_QACC_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_u16_qacc_ld_incp_2p2_p $qu, $rs1, $qx, $qy, $sel16, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_u16_qacc_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$sel16, timm:$qu))]>;

def ESP_VSMULAS_U8_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qx, QR:$qy, sat:$sat, select_16:$sel16),
                   "esp.vsmulas.u8.qacc\t $qx, $qy, $sel16, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{20} = sat{0};
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSMULAS_U8_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, select_16:$sel16),
				"!esp_vsmulas_u8_qacc_2p2_p $qx, $qy, $sel16, $sat",
			[(int_riscv_esp_vsmulas_u8_qacc_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$sel16)]>;

def ESP_VSMULAS_U8_QACC_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, select_16:$sel16),
                   "esp.vsmulas.u8.qacc.ld.incp\t $qu, $rs1, $qx, $qy, $sel16, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{23} = sat{0};
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
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSMULAS_U8_QACC_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, select_16:$sel16, imm8:$qu),
				"!esp_vsmulas_u8_qacc_ld_incp_2p2_p $qu, $rs1, $qx, $qy, $sel16, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsmulas_u8_qacc_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$sel16, timm:$qu))]>;

def ESP_CMUL_S16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.s16\t $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = sat{0};
  let Inst{16-15} = sel4{1-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = rm{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_CMUL_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz),
				"!esp_cmul_s16_2p2_p $qz, $qx, $qy, $sel4, $sat, $rm",
			[(int_riscv_esp_cmul_s16_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$sel4, timm:$rm, timm:$qz)]>;

def ESP_CMUL_S16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.s16.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
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
  let Inst{23} = 1;
  let Inst{22} = sat{0};
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = rm{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14-13} = rm{1-0};
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
def ESP_CMUL_S16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz, imm8:$qu),
				"!esp_cmul_s16_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$sel4, timm:$rm, timm:$qz, timm:$qu))]>;

def ESP_CMUL_S16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.s16.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = rm{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14-13} = rm{1-0};
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
def ESP_CMUL_S16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz),
				"!esp_cmul_s16_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$sel4, timm:$rm, timm:$qz))]>;

def ESP_CMUL_S8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.s8\t $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = sat{0};
  let Inst{16-15} = sel4{1-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = rm{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_CMUL_S8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz),
				"!esp_cmul_s8_2p2_p $qz, $qx, $qy, $sel4, $sat, $rm",
			[(int_riscv_esp_cmul_s8_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$sel4, timm:$rm, timm:$qz)]>;

def ESP_CMUL_S8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.s8.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = rm{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14-13} = rm{1-0};
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
def ESP_CMUL_S8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz, imm8:$qu),
				"!esp_cmul_s8_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$sel4, timm:$rm, timm:$qz, timm:$qu))]>;

def ESP_CMUL_S8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.s8.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = rm{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14-13} = rm{1-0};
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
def ESP_CMUL_S8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz),
				"!esp_cmul_s8_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_s8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$sel4, timm:$rm, timm:$qz))]>;

def ESP_CMUL_U16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.u16\t $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = sat{0};
  let Inst{16-15} = sel4{1-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = rm{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_CMUL_U16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz),
				"!esp_cmul_u16_2p2_p $qz, $qx, $qy, $sel4, $sat, $rm",
			[(int_riscv_esp_cmul_u16_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$sel4, timm:$rm, timm:$qz)]>;

def ESP_CMUL_U16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.u16.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = rm{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14-13} = rm{1-0};
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
def ESP_CMUL_U16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz, imm8:$qu),
				"!esp_cmul_u16_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$sel4, timm:$rm, timm:$qz, timm:$qu))]>;

def ESP_CMUL_U16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.u16.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
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
  let Inst{23} = 0;
  let Inst{22} = sat{0};
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = rm{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14-13} = rm{1-0};
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
def ESP_CMUL_U16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz),
				"!esp_cmul_u16_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$sel4, timm:$rm, timm:$qz))]>;

def ESP_CMUL_U8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.u8\t $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = sat{0};
  let Inst{16-15} = sel4{1-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = rm{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_CMUL_U8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz),
				"!esp_cmul_u8_2p2_p $qz, $qx, $qy, $sel4, $sat, $rm",
			[(int_riscv_esp_cmul_u8_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$sel4, timm:$rm, timm:$qz)]>;

def ESP_CMUL_U8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.u8.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = rm{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14-13} = rm{1-0};
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
def ESP_CMUL_U8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz, imm8:$qu),
				"!esp_cmul_u8_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$sel4, timm:$rm, timm:$qz, timm:$qu))]>;

def ESP_CMUL_U8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm),
                   "esp.cmul.u8.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
  bits<2> sel4;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-20} = sel4{1-0};
  let Inst{19} = rm{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14-13} = rm{1-0};
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
def ESP_CMUL_U8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, select_4:$sel4, rm:$rm, imm8:$qz),
				"!esp_cmul_u8_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sel4, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_cmul_u8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$sel4, timm:$rm, timm:$qz))]>;

def ESP_MAX_S16_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.s16.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
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
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_S16_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_s16_a_2p2_p $qw, $rd",
			[(int_riscv_esp_max_s16_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_S32_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.s32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
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
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_S32_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_s32_a_2p2_p $qw, $rd",
			[(int_riscv_esp_max_s32_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_S8_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.s8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
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
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_S8_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_s8_a_2p2_p $qw, $rd",
			[(int_riscv_esp_max_s8_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_U16_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
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
  let Inst{28} = 0;
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
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_U16_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_u16_a_2p2_p $qw, $rd",
			[(int_riscv_esp_max_u16_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_U32_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.u32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
  let Inst{28} = 0;
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
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_U32_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_u32_a_2p2_p $qw, $rd",
			[(int_riscv_esp_max_u32_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MAX_U8_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.max.u8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
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
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MAX_U8_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_max_u8_a_2p2_p $qw, $rd",
			[(int_riscv_esp_max_u8_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_S16_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.s16.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
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
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_S16_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_s16_a_2p2_p $qw, $rd",
			[(int_riscv_esp_min_s16_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_S32_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.s32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 1;
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
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_S32_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_s32_a_2p2_p $qw, $rd",
			[(int_riscv_esp_min_s32_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_S8_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.s8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
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
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_S8_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_s8_a_2p2_p $qw, $rd",
			[(int_riscv_esp_min_s8_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_U16_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_U16_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_u16_a_2p2_p $qw, $rd",
			[(int_riscv_esp_min_u16_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_U32_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.u32.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 1;
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
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_U32_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_u32_a_2p2_p $qw, $rd",
			[(int_riscv_esp_min_u32_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_MIN_U8_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qw),
                   "esp.min.u8.a\t $qw, $rd", []> 
{
  bits<3> qw;
  bits<5> rd;
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
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MIN_U8_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, GPRPIE:$rd),
				"!esp_min_u8_a_2p2_p $qw, $rd",
			[(int_riscv_esp_min_u8_a_2p2 timm:$qw, GPRPIE:$rd)]>;

def ESP_VABS_16_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qy),
                   "esp.vabs.16\t $qv, $qy", []> 
{
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VABS_16_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qv),
				"!esp_vabs_16_2p2_p $qv, $qy",
			[(int_riscv_esp_vabs_16_2p2 timm:$qy, timm:$qv)]>;

def ESP_VABS_32_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qy),
                   "esp.vabs.32\t $qv, $qy", []> 
{
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VABS_32_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qv),
				"!esp_vabs_32_2p2_p $qv, $qy",
			[(int_riscv_esp_vabs_32_2p2 timm:$qy, timm:$qv)]>;

def ESP_VABS_8_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qy),
                   "esp.vabs.8\t $qv, $qy", []> 
{
  bits<3> qy;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
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
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VABS_8_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qv),
				"!esp_vabs_8_2p2_p $qv, $qy",
			[(int_riscv_esp_vabs_8_2p2 timm:$qy, timm:$qv)]>;

def ESP_VADD_S16_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vadd.s16\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vadd_s16_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vadd_s16_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VADD_S16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.s16.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vadd_s16_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VADD_S16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.s16.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vadd_s16_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VADD_S32_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vadd.s32\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{17} = sat{0};
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_S32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vadd_s32_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vadd_s32_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VADD_S32_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.s32.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S32_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vadd_s32_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s32_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VADD_S32_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.s32.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S32_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vadd_s32_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s32_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VADD_S8_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vadd.s8\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_S8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vadd_s8_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vadd_s8_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VADD_S8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.s8.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vadd_s8_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VADD_S8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.s8.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_S8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vadd_s8_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_s8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VADD_U16_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vadd.u16\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_U16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vadd_u16_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vadd_u16_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VADD_U16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.u16.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vadd_u16_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VADD_U16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.u16.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vadd_u16_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VADD_U32_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vadd.u32\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{17} = sat{0};
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_U32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vadd_u32_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vadd_u32_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VADD_U32_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.u32.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U32_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vadd_u32_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u32_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VADD_U32_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.u32.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U32_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vadd_u32_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u32_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VADD_U8_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vadd.u8\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VADD_U8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vadd_u8_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vadd_u8_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VADD_U8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.u8.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vadd_u8_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VADD_U8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vadd.u8.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VADD_U8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vadd_u8_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vadd_u8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VCLAMP_S16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, select_16:$sel16),
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
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18-15} = sel16{3-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCLAMP_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, select_16:$sel16, imm8:$qz),
				"!esp_vclamp_s16_2p2_p $qz, $qx, $sel16",
			[(int_riscv_esp_vclamp_s16_2p2 timm:$qx, timm:$sel16, timm:$qz)]>;

def ESP_VMAX_S16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_s16_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_s16_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_S16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_S16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_s16_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_S16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_S16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_s16_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMAX_S32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_S32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_s32_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_s32_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_S32_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_S32_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_s32_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s32_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_S32_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_S32_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_s32_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s32_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMAX_S8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_S8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_s8_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_s8_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_S8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_S8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_s8_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_S8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_S8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_s8_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_s8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMAX_U16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_U16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_u16_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_u16_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_U16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_U16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_u16_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_U16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_U16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_u16_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMAX_U32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_U32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_u32_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_u32_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_U32_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_U32_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_u32_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u32_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_U32_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_U32_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_u32_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u32_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMAX_U8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
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
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMAX_U8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmax_u8_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmax_u8_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMAX_U8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_U8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmax_u8_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMAX_U8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMAX_U8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmax_u8_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmax_u8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_S16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_s16_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_s16_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_S16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMIN_S16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_s16_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_S16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{25} = 1;
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
def ESP_VMIN_S16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_s16_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_S32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_S32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_s32_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_s32_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_S32_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMIN_S32_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_s32_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s32_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_S32_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{25} = 1;
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
def ESP_VMIN_S32_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_s32_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s32_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_S8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_S8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_s8_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_s8_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_S8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMIN_S8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_s8_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_S8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{25} = 1;
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
def ESP_VMIN_S8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_s8_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_s8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_U16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_U16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_u16_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_u16_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_U16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMIN_U16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_u16_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_U16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMIN_U16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_u16_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_U32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_U32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_u32_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_u32_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_U32_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMIN_U32_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_u32_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u32_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_U32_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{25} = 1;
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
def ESP_VMIN_U32_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_u32_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u32_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMIN_U8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMIN_U8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vmin_u8_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vmin_u8_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VMIN_U8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1),
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
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMIN_U8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, imm8:$qz, imm8:$qu),
				"!esp_vmin_u8_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$qz, timm:$qu))]>;

def ESP_VMIN_U8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1),
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
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
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
def ESP_VMIN_U8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, imm8:$qz),
				"!esp_vmin_u8_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmin_u8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$qz))]>;

def ESP_VMUL_S16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, sat:$sat, rm:$rm),
                   "esp.vmul.s16\t $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
  bits<3> rm;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 1;
  let Inst{16} = sat{0};
  let Inst{15} = rm{2};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-11} = rm{1-0};
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, rm:$rm, imm8:$qz),
				"!esp_vmul_s16_2p2_p $qz, $qx, $qy, $sat, $rm",
			[(int_riscv_esp_vmul_s16_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$rm, timm:$qz)]>;

def ESP_VMUL_S16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.vmul.s16.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
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
  let Inst{23} = 1;
  let Inst{22} = sat{0};
  let Inst{21-19} = rm{2-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
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
def ESP_VMUL_S16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qz, imm8:$qu),
				"!esp_vmul_s16_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qz, timm:$qu))]>;

def ESP_VMUL_S16_S8XS8_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qv), (ins QR:$qx, QR:$qy, rm:$rm),
                   "esp.vmul.s16.s8xs8\t $qz, $qv, $qx, $qy, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> rm;
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
  let Inst{18-16} = rm{2-0};
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_S16_S8XS8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, rm:$rm, imm8:$qz, imm8:$qv),
				"!esp_vmul_s16_s8xs8_2p2_p $qz, $qv, $qx, $qy, $rm",
			[(int_riscv_esp_vmul_s16_s8xs8_2p2 timm:$qx, timm:$qy, timm:$rm, timm:$qz, timm:$qv)]>;

def ESP_VMUL_S16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.vmul.s16.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-19} = rm{2-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
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
def ESP_VMUL_S16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qz),
				"!esp_vmul_s16_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qz))]>;

def ESP_VMUL_S32_S16XS16_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qv), (ins QR:$qx, QR:$qy, rm:$rm),
                   "esp.vmul.s32.s16xs16\t $qz, $qv, $qx, $qy, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> rm;
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
  let Inst{18-16} = rm{2-0};
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_S32_S16XS16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, rm:$rm, imm8:$qz, imm8:$qv),
				"!esp_vmul_s32_s16xs16_2p2_p $qz, $qv, $qx, $qy, $rm",
			[(int_riscv_esp_vmul_s32_s16xs16_2p2 timm:$qx, timm:$qy, timm:$rm, timm:$qz, timm:$qv)]>;

def ESP_VMUL_S8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, sat:$sat, rm:$rm),
                   "esp.vmul.s8\t $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
  bits<3> rm;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = sat{0};
  let Inst{15} = rm{2};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-11} = rm{1-0};
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_S8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, rm:$rm, imm8:$qz),
				"!esp_vmul_s8_2p2_p $qz, $qx, $qy, $sat, $rm",
			[(int_riscv_esp_vmul_s8_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$rm, timm:$qz)]>;

def ESP_VMUL_S8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.vmul.s8.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-19} = rm{2-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
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
def ESP_VMUL_S8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qz, imm8:$qu),
				"!esp_vmul_s8_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qz, timm:$qu))]>;

def ESP_VMUL_S8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.vmul.s8.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-19} = rm{2-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
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
def ESP_VMUL_S8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qz),
				"!esp_vmul_s8_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_s8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qz))]>;

def ESP_VMUL_U16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, sat:$sat, rm:$rm),
                   "esp.vmul.u16\t $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
  bits<3> rm;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = sat{0};
  let Inst{15} = rm{2};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-11} = rm{1-0};
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_U16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, rm:$rm, imm8:$qz),
				"!esp_vmul_u16_2p2_p $qz, $qx, $qy, $sat, $rm",
			[(int_riscv_esp_vmul_u16_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$rm, timm:$qz)]>;

def ESP_VMUL_U16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.vmul.u16.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-19} = rm{2-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
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
def ESP_VMUL_U16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qz, imm8:$qu),
				"!esp_vmul_u16_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qz, timm:$qu))]>;

def ESP_VMUL_U16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.vmul.u16.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
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
  let Inst{23} = 0;
  let Inst{22} = sat{0};
  let Inst{21-19} = rm{2-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
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
def ESP_VMUL_U16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qz),
				"!esp_vmul_u16_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qz))]>;

def ESP_VMUL_U8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy, sat:$sat, rm:$rm),
                   "esp.vmul.u8\t $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
  bits<3> rm;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = sat{0};
  let Inst{15} = rm{2};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-11} = rm{1-0};
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VMUL_U8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, rm:$rm, imm8:$qz),
				"!esp_vmul_u8_2p2_p $qz, $qx, $qy, $sat, $rm",
			[(int_riscv_esp_vmul_u8_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$rm, timm:$qz)]>;

def ESP_VMUL_U8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.vmul.u8.ld.incp\t $qu, $rs1, $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-19} = rm{2-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
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
def ESP_VMUL_U8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qz, imm8:$qu),
				"!esp_vmul_u8_ld_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qz, timm:$qu))]>;

def ESP_VMUL_U8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.vmul.u8.st.incp\t $qu, $rs1, $qz, $qx, $qy, $sat, $rm", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
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
  let Inst{22} = sat{0};
  let Inst{21-19} = rm{2-0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
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
def ESP_VMUL_U8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qz),
				"!esp_vmul_u8_st_incp_2p2_p $qu, $rs1, $qz, $qx, $qy, $sat, $rm",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vmul_u8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qz))]>;

def ESP_VPRELU_S16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins GPRPIE:$rs1, QR:$qx, QR:$qy, sat:$sat, rm:$rm),
                   "esp.vprelu.s16\t $qz, $qy, $qx, $rs1, $sat, $rm", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
  bits<3> rm;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = sat{0};
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = rm{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VPRELU_S16_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qy, sat:$sat, rm:$rm, imm8:$qz),
				"!esp_vprelu_s16_2p2_p $qz, $qy, $qx, $rs1, $sat, $rm",
			[(int_riscv_esp_vprelu_s16_2p2 GPRPIE:$rs1, timm:$qx, timm:$qy, timm:$sat, timm:$rm, timm:$qz)]>;

def ESP_VPRELU_S8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins GPRPIE:$rs1, QR:$qx, QR:$qy, sat:$sat, rm:$rm),
                   "esp.vprelu.s8\t $qz, $qy, $qx, $rs1, $sat, $rm", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
  bits<3> rm;
  bits<3> qz;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = sat{0};
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = rm{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VPRELU_S8_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, imm8:$qy, sat:$sat, rm:$rm, imm8:$qz),
				"!esp_vprelu_s8_2p2_p $qz, $qy, $qx, $rs1, $sat, $rm",
			[(int_riscv_esp_vprelu_s8_2p2 GPRPIE:$rs1, timm:$qx, timm:$qy, timm:$sat, timm:$rm, timm:$qz)]>;

def ESP_VRELU_S16_2P2: Esp32V2P2Inst<(outs QR:$qyr), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy, sat:$sat, rm:$rm),
                   "esp.vrelu.s16\t $qy, $rs2, $rs1, $sat, $rm", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qy;
  bits<1> sat;
  bits<3> rm;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qyr = $qy";
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = sat{0};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = rm{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VRELU_S16_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy, sat:$sat, rm:$rm),
				"!esp_vrelu_s16_2p2_p $qy, $rs2, $rs1, $sat, $rm",
			[(int_riscv_esp_vrelu_s16_2p2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy, timm:$sat, timm:$rm)]>;

def ESP_VRELU_S8_2P2: Esp32V2P2Inst<(outs QR:$qyr), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy, sat:$sat, rm:$rm),
                   "esp.vrelu.s8\t $qy, $rs2, $rs1, $sat, $rm", []> 
{
  bits<5> rs1;
  bits<5> rs2;
  bits<3> qy;
  bits<1> sat;
  bits<3> rm;
  bits<3> qyr;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$qyr = $qy";
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = sat{0};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = rm{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VRELU_S8_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy, sat:$sat, rm:$rm),
				"!esp_vrelu_s8_2p2_p $qy, $rs2, $rs1, $sat, $rm",
			[(int_riscv_esp_vrelu_s8_2p2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy, timm:$sat, timm:$rm)]>;

def ESP_VSADDS_S16_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx, sat:$sat),
                   "esp.vsadds.s16\t $qv, $qx, $rs1, $sat", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<1> sat;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = sat{0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSADDS_S16_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, sat:$sat, imm8:$qv),
				"!esp_vsadds_s16_2p2_p $qv, $qx, $rs1, $sat",
			[(int_riscv_esp_vsadds_s16_2p2 GPRPIE:$rs1, timm:$qx, timm:$sat, timm:$qv)]>;

def ESP_VSADDS_S8_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx, sat:$sat),
                   "esp.vsadds.s8\t $qv, $qx, $rs1, $sat", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<1> sat;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = sat{0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSADDS_S8_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, sat:$sat, imm8:$qv),
				"!esp_vsadds_s8_2p2_p $qv, $qx, $rs1, $sat",
			[(int_riscv_esp_vsadds_s8_2p2 GPRPIE:$rs1, timm:$qx, timm:$sat, timm:$qv)]>;

def ESP_VSADDS_U16_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx, sat:$sat),
                   "esp.vsadds.u16\t $qv, $qx, $rs1, $sat", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<1> sat;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = sat{0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSADDS_U16_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, sat:$sat, imm8:$qv),
				"!esp_vsadds_u16_2p2_p $qv, $qx, $rs1, $sat",
			[(int_riscv_esp_vsadds_u16_2p2 GPRPIE:$rs1, timm:$qx, timm:$sat, timm:$qv)]>;

def ESP_VSADDS_U8_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx, sat:$sat),
                   "esp.vsadds.u8\t $qv, $qx, $rs1, $sat", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<1> sat;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = sat{0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSADDS_U8_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, sat:$sat, imm8:$qv),
				"!esp_vsadds_u8_2p2_p $qv, $qx, $rs1, $sat",
			[(int_riscv_esp_vsadds_u8_2p2 GPRPIE:$rs1, timm:$qx, timm:$sat, timm:$qv)]>;

def ESP_VSAT_S16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
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
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_S16_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_s16_2p2_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_s16_2p2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_S32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
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
  let Inst{27} = 0;
  let Inst{26} = 1;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_S32_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_s32_2p2_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_s32_2p2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_S8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
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
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_S8_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_s8_2p2_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_s8_2p2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_U16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
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
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_U16_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_u16_2p2_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_u16_2p2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_U32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
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
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_U32_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_u32_2p2_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_u32_2p2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSAT_U8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qx),
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
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSAT_U8_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qx, imm8:$qz),
				"!esp_vsat_u8_2p2_p $qz, $qx, $rs1, $rs2",
			[(int_riscv_esp_vsat_u8_2p2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qx, timm:$qz)]>;

def ESP_VSSUBS_S16_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx, sat:$sat),
                   "esp.vssubs.s16\t $qv, $qx, $rs1, $sat", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<1> sat;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = sat{0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSSUBS_S16_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, sat:$sat, imm8:$qv),
				"!esp_vssubs_s16_2p2_p $qv, $qx, $rs1, $sat",
			[(int_riscv_esp_vssubs_s16_2p2 GPRPIE:$rs1, timm:$qx, timm:$sat, timm:$qv)]>;

def ESP_VSSUBS_S8_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx, sat:$sat),
                   "esp.vssubs.s8\t $qv, $qx, $rs1, $sat", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<1> sat;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = sat{0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSSUBS_S8_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, sat:$sat, imm8:$qv),
				"!esp_vssubs_s8_2p2_p $qv, $qx, $rs1, $sat",
			[(int_riscv_esp_vssubs_s8_2p2 GPRPIE:$rs1, timm:$qx, timm:$sat, timm:$qv)]>;

def ESP_VSSUBS_U16_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx, sat:$sat),
                   "esp.vssubs.u16\t $qv, $qx, $rs1, $sat", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<1> sat;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = sat{0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSSUBS_U16_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, sat:$sat, imm8:$qv),
				"!esp_vssubs_u16_2p2_p $qv, $qx, $rs1, $sat",
			[(int_riscv_esp_vssubs_u16_2p2 GPRPIE:$rs1, timm:$qx, timm:$sat, timm:$qv)]>;

def ESP_VSSUBS_U8_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins GPRPIE:$rs1, QR:$qx, sat:$sat),
                   "esp.vssubs.u8\t $qv, $qx, $rs1, $sat", []> 
{
  bits<5> rs1;
  bits<3> qx;
  bits<1> sat;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = sat{0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSSUBS_U8_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qx, sat:$sat, imm8:$qv),
				"!esp_vssubs_u8_2p2_p $qv, $qx, $rs1, $sat",
			[(int_riscv_esp_vssubs_u8_2p2 GPRPIE:$rs1, timm:$qx, timm:$sat, timm:$qv)]>;

def ESP_VSUB_S16_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vsub.s16\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vsub_s16_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vsub_s16_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VSUB_S16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.s16.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vsub_s16_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VSUB_S16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.s16.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vsub_s16_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VSUB_S32_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vsub.s32\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{17} = sat{0};
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_S32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vsub_s32_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vsub_s32_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VSUB_S32_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.s32.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S32_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vsub_s32_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s32_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VSUB_S32_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.s32.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S32_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vsub_s32_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s32_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VSUB_S8_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vsub.s8\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_S8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vsub_s8_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vsub_s8_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VSUB_S8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.s8.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vsub_s8_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VSUB_S8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.s8.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_S8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vsub_s8_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_s8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VSUB_U16_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vsub.u16\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_U16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vsub_u16_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vsub_u16_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VSUB_U16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.u16.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vsub_u16_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u16_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VSUB_U16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.u16.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vsub_u16_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VSUB_U32_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vsub.u32\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{17} = sat{0};
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_U32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vsub_u32_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vsub_u32_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VSUB_U32_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.u32.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U32_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vsub_u32_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u32_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VSUB_U32_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.u32.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U32_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vsub_u32_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u32_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_VSUB_U8_2P2: Esp32V2P2Inst<(outs QR:$qv), (ins QR:$qx, QR:$qy, sat:$sat),
                   "esp.vsub.u8\t $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sat;
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
  let Inst{16} = sat{0};
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSUB_U8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, sat:$sat, imm8:$qv),
				"!esp_vsub_u8_2p2_p $qv, $qx, $qy, $sat",
			[(int_riscv_esp_vsub_u8_2p2 timm:$qx, timm:$qy, timm:$sat, timm:$qv)]>;

def ESP_VSUB_U8_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, QR:$qu, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.u8.ld.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U8_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, imm8:$qv, imm8:$qu),
				"!esp_vsub_u8_ld_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u8_ld_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$qv, timm:$qu))]>;

def ESP_VSUB_U8_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat),
                   "esp.vsub.u8.st.incp\t $qu, $rs1, $qv, $qx, $qy, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{19} = sat{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VSUB_U8_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, imm8:$qv),
				"!esp_vsub_u8_st_incp_2p2_p $qu, $rs1, $qv, $qx, $qy, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vsub_u8_st_incp_2p2 timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$qv))]>;

def ESP_ADDX2_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1, GPRPIE:$rs2),
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
def ESP_ADDX2_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd),
				"!esp_addx2_2p2_p $rd, $rs1, $rs2",
			[(int_riscv_esp_addx2_2p2 GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd)]>;

def ESP_ADDX4_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1, GPRPIE:$rs2),
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
def ESP_ADDX4_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd),
				"!esp_addx4_2p2_p $rd, $rs1, $rs2",
			[(int_riscv_esp_addx4_2p2 GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd)]>;

def ESP_SAT_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs2r), (ins GPRPIE:$rd, GPRPIE:$rs1, GPRPIE:$rs2),
                   "esp.sat\t $rs2, $rd, $rs1", []> 
{
  bits<5> rd;
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rs2r;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs2r = $rs2";
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24-20} = rs2{4-0};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
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

let usesCustomInserter = 1, Constraints = "$rs2r = $rs2" in 
def ESP_SAT_2P2_P : PseudoEsp<(outs GPRPIE:$rs2r), (ins GPRPIE:$rd, GPRPIE:$rs1, GPRPIE:$rs2),
				"!esp_sat_2p2_p $rs2, $rd, $rs1",
			[(set GPRPIE:$rs2r, (int_riscv_esp_sat_2p2 GPRPIE:$rd, GPRPIE:$rs1, GPRPIE:$rs2))]>;

def ESP_SUBX2_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1, GPRPIE:$rs2),
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
def ESP_SUBX2_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd),
				"!esp_subx2_2p2_p $rd, $rs1, $rs2",
			[(int_riscv_esp_subx2_2p2 GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd)]>;

def ESP_SUBX4_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1, GPRPIE:$rs2),
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
def ESP_SUBX4_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd),
				"!esp_subx4_2p2_p $rd, $rs1, $rs2",
			[(int_riscv_esp_subx4_2p2 GPRPIE:$rs1, GPRPIE:$rs2, GPRPIE:$rd)]>;

def ESP_ANDQ_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{24} = 1;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_ANDQ_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_andq_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_andq_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_NOTQ_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx),
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
  let Inst{24} = 1;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_NOTQ_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qz),
				"!esp_notq_2p2_p $qz, $qx",
			[(int_riscv_esp_notq_2p2 timm:$qx, timm:$qz)]>;

def ESP_ORQ_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{24} = 1;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_ORQ_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_orq_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_orq_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_XORQ_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{24} = 1;
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
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_XORQ_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_xorq_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_xorq_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_S16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_s16_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_s16_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_S32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_S32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_s32_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_s32_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_S8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_S8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_s8_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_s8_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_U16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_U16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_u16_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_u16_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_U32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
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
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_U32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_u32_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_u32_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_EQ_U8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_EQ_U8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_eq_u8_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_eq_u8_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_S16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_s16_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_s16_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_S32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_S32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_s32_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_s32_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_S8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_S8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_s8_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_s8_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_U16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_U16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_u16_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_u16_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_U32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_U32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_u32_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_u32_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_GT_U8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_GT_U8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_gt_u8_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_gt_u8_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_S16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_s16_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_s16_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_S32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_S32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_s32_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_s32_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_S8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = 0;
  let Inst{17} = 1;
  let Inst{16} = 0;
  let Inst{15} = 1;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_S8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_s8_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_s8_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_U16_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_U16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_u16_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_u16_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_U32_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{11} = 1;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_U32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_u32_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_u32_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_VCMP_LT_U8_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qx, QR:$qy),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
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
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VCMP_LT_U8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qz),
				"!esp_vcmp_lt_u8_2p2_p $qz, $qx, $qy",
			[(int_riscv_esp_vcmp_lt_u8_2p2 timm:$qx, timm:$qy, timm:$qz)]>;

def ESP_MOV_S16_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qu),
                   "esp.mov.s16.qacc\t $qu", []> 
{
  bits<3> qu;
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
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOV_S16_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qu),
				"!esp_mov_s16_qacc_2p2_p $qu",
			[(int_riscv_esp_mov_s16_qacc_2p2 timm:$qu)]>;

def ESP_MOV_S8_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qu),
                   "esp.mov.s8.qacc\t $qu", []> 
{
  bits<3> qu;
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
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOV_S8_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qu),
				"!esp_mov_s8_qacc_2p2_p $qu",
			[(int_riscv_esp_mov_s8_qacc_2p2 timm:$qu)]>;

def ESP_MOV_U16_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qu),
                   "esp.mov.u16.qacc\t $qu", []> 
{
  bits<3> qu;
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
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOV_U16_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qu),
				"!esp_mov_u16_qacc_2p2_p $qu",
			[(int_riscv_esp_mov_u16_qacc_2p2 timm:$qu)]>;

def ESP_MOV_U8_QACC_2P2: Esp32V2P2Inst<(outs), (ins QR:$qu),
                   "esp.mov.u8.qacc\t $qu", []> 
{
  bits<3> qu;
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOV_U8_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qu),
				"!esp_mov_u8_qacc_2p2_p $qu",
			[(int_riscv_esp_mov_u8_qacc_2p2 timm:$qu)]>;

def ESP_MOVI_16_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qy, select_16:$sel16),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_16_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, select_16:$sel16, GPRPIE:$rd),
				"!esp_movi_16_a_2p2_p $qy, $rd, $sel16",
			[(int_riscv_esp_movi_16_a_2p2 timm:$qy, timm:$sel16, GPRPIE:$rd)]>;

def ESP_MOVI_16_Q_2P2: Esp32V2P2Inst<(outs QR:$qy), (ins GPRPIE:$rs1, select_16:$sel16),
                   "esp.movi.16.q\t $qy, $rs1, $sel16", []> 
{
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qy;
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
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10-7} = sel16{3-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_16_Q_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, select_16:$sel16, imm8:$qy),
				"!esp_movi_16_q_2p2_p $qy, $rs1, $sel16",
			[(int_riscv_esp_movi_16_q_2p2 GPRPIE:$rs1, timm:$sel16, timm:$qy)]>;

def ESP_MOVI_32_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qy, select_4:$sel4),
                   "esp.movi.32.a\t $qy, $rd, $sel4", []> 
{
  bits<3> qy;
  bits<2> sel4;
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
  let Inst{22} = sel4{1};
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = sel4{0};
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_32_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, select_4:$sel4, GPRPIE:$rd),
				"!esp_movi_32_a_2p2_p $qy, $rd, $sel4",
			[(int_riscv_esp_movi_32_a_2p2 timm:$qy, timm:$sel4, GPRPIE:$rd)]>;

def ESP_MOVI_32_Q_2P2: Esp32V2P2Inst<(outs QR:$qy), (ins GPRPIE:$rs1, select_4:$sel4),
                   "esp.movi.32.q\t $qy, $rs1, $sel4", []> 
{
  bits<5> rs1;
  bits<2> sel4;
  bits<3> qy;
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
  let Inst{21} = 1;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10-9} = sel4{1-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_32_Q_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, select_4:$sel4, imm8:$qy),
				"!esp_movi_32_q_2p2_p $qy, $rs1, $sel4",
			[(int_riscv_esp_movi_32_q_2p2 GPRPIE:$rs1, timm:$sel4, timm:$qy)]>;

def ESP_MOVI_8_A_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins QR:$qy, select_16:$sel16),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_8_A_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, select_16:$sel16, GPRPIE:$rd),
				"!esp_movi_8_a_2p2_p $qy, $rd, $sel16",
			[(int_riscv_esp_movi_8_a_2p2 timm:$qy, timm:$sel16, GPRPIE:$rd)]>;

def ESP_MOVI_8_Q_2P2: Esp32V2P2Inst<(outs QR:$qy), (ins GPRPIE:$rs1, select_16:$sel16),
                   "esp.movi.8.q\t $qy, $rs1, $sel16", []> 
{
  bits<5> rs1;
  bits<4> sel16;
  bits<3> qy;
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
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10-7} = sel16{3-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVI_8_Q_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, select_16:$sel16, imm8:$qy),
				"!esp_movi_8_q_2p2_p $qy, $rs1, $sel16",
			[(int_riscv_esp_movi_8_q_2p2 GPRPIE:$rs1, timm:$sel16, timm:$qy)]>;

def ESP_MOVX_R_CFG_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins),
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
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_CFG_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_cfg_2p2_p $rd",
			[(int_riscv_esp_movx_r_cfg_2p2 GPRPIE:$rd)]>;

def ESP_MOVX_R_FFT_BIT_WIDTH_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins),
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
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_FFT_BIT_WIDTH_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_fft_bit_width_2p2_p $rd",
			[(int_riscv_esp_movx_r_fft_bit_width_2p2 GPRPIE:$rd)]>;

def ESP_MOVX_R_PERF_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1),
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
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_PERF_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rd),
				"!esp_movx_r_perf_2p2_p $rd, $rs1",
			[(int_riscv_esp_movx_r_perf_2p2 GPRPIE:$rs1, GPRPIE:$rd)]>;

def ESP_MOVX_R_SAR_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins),
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
  let Inst{22} = 1;
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
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_SAR_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_sar_2p2_p $rd",
			[(int_riscv_esp_movx_r_sar_2p2 GPRPIE:$rd)]>;

def ESP_MOVX_R_SAR_BYTES_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins),
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
  let Inst{22} = 1;
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
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_SAR_BYTES_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_sar_bytes_2p2_p $rd",
			[(int_riscv_esp_movx_r_sar_bytes_2p2 GPRPIE:$rd)]>;

def ESP_MOVX_R_XACC_H_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins),
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
  let Inst{22} = 1;
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
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_XACC_H_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_xacc_h_2p2_p $rd",
			[(int_riscv_esp_movx_r_xacc_h_2p2 GPRPIE:$rd)]>;

def ESP_MOVX_R_XACC_L_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins),
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
  let Inst{22} = 1;
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
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_R_XACC_L_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rd),
				"!esp_movx_r_xacc_l_2p2_p $rd",
			[(int_riscv_esp_movx_r_xacc_l_2p2 GPRPIE:$rd)]>;

def ESP_MOVX_W_CFG_2P2: Esp32V2P2Inst<(outs), (ins GPRPIE:$rs1),
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
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_CFG_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_cfg_2p2_p $rs1",
			[(int_riscv_esp_movx_w_cfg_2p2 GPRPIE:$rs1)]>;

def ESP_MOVX_W_FFT_BIT_WIDTH_2P2: Esp32V2P2Inst<(outs), (ins GPRPIE:$rs1),
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
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_FFT_BIT_WIDTH_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_fft_bit_width_2p2_p $rs1",
			[(int_riscv_esp_movx_w_fft_bit_width_2p2 GPRPIE:$rs1)]>;

def ESP_MOVX_W_PERF_2P2: Esp32V2P2Inst<(outs), (ins GPRPIE:$rs1),
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
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_PERF_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_perf_2p2_p $rs1",
			[(int_riscv_esp_movx_w_perf_2p2 GPRPIE:$rs1)]>;

def ESP_MOVX_W_SAR_2P2: Esp32V2P2Inst<(outs), (ins GPRPIE:$rs1),
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
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_SAR_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_sar_2p2_p $rs1",
			[(int_riscv_esp_movx_w_sar_2p2 GPRPIE:$rs1)]>;

def ESP_MOVX_W_SAR_BYTES_2P2: Esp32V2P2Inst<(outs), (ins GPRPIE:$rs1),
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
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_SAR_BYTES_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_sar_bytes_2p2_p $rs1",
			[(int_riscv_esp_movx_w_sar_bytes_2p2 GPRPIE:$rs1)]>;

def ESP_MOVX_W_XACC_H_2P2: Esp32V2P2Inst<(outs), (ins GPRPIE:$rs1),
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
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_XACC_H_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_xacc_h_2p2_p $rs1",
			[(int_riscv_esp_movx_w_xacc_h_2p2 GPRPIE:$rs1)]>;

def ESP_MOVX_W_XACC_L_2P2: Esp32V2P2Inst<(outs), (ins GPRPIE:$rs1),
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
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_MOVX_W_XACC_L_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1),
				"!esp_movx_w_xacc_l_2p2_p $rs1",
			[(int_riscv_esp_movx_w_xacc_l_2p2 GPRPIE:$rs1)]>;

def ESP_VEXT_S16_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VEXT_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_s16_2p2_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_s16_2p2 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VEXT_S8_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VEXT_S8_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_s8_2p2_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_s8_2p2 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VEXT_U16_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VEXT_U16_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_u16_2p2_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_u16_2p2 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VEXT_U8_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qv), (ins QR:$qw),
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VEXT_U8_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, imm8:$qz, imm8:$qv),
				"!esp_vext_u8_2p2_p $qz, $qv, $qw",
			[(int_riscv_esp_vext_u8_2p2 timm:$qw, timm:$qz, timm:$qv)]>;

def ESP_VUNZIP_16_2P2: Esp32V2P2Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VUNZIP_16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vunzip_16_2p2_p $qx, $qy",
			[(int_riscv_esp_vunzip_16_2p2 timm:$qx, timm:$qy)]>;

def ESP_VUNZIP_32_2P2: Esp32V2P2Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
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
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VUNZIP_32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vunzip_32_2p2_p $qx, $qy",
			[(int_riscv_esp_vunzip_32_2p2 timm:$qx, timm:$qy)]>;

def ESP_VUNZIP_8_2P2: Esp32V2P2Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VUNZIP_8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vunzip_8_2p2_p $qx, $qy",
			[(int_riscv_esp_vunzip_8_2p2 timm:$qx, timm:$qy)]>;

def ESP_VUNZIPT_16_2P2: Esp32V2P2Inst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VUNZIPT_16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vunzipt_16_2p2_p $qx, $qy, $qw",
			[(int_riscv_esp_vunzipt_16_2p2 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_VUNZIPT_8_2P2: Esp32V2P2Inst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VUNZIPT_8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vunzipt_8_2p2_p $qx, $qy, $qw",
			[(int_riscv_esp_vunzipt_8_2p2 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_VZIP_16_2P2: Esp32V2P2Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VZIP_16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vzip_16_2p2_p $qx, $qy",
			[(int_riscv_esp_vzip_16_2p2 timm:$qx, timm:$qy)]>;

def ESP_VZIP_32_2P2: Esp32V2P2Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
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
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VZIP_32_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vzip_32_2p2_p $qx, $qy",
			[(int_riscv_esp_vzip_32_2p2 timm:$qx, timm:$qy)]>;

def ESP_VZIP_8_2P2: Esp32V2P2Inst<(outs QR:$qxr, QR:$qyr), (ins QR:$qx, QR:$qy),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VZIP_8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy),
				"!esp_vzip_8_2p2_p $qx, $qy",
			[(int_riscv_esp_vzip_8_2p2 timm:$qx, timm:$qy)]>;

def ESP_VZIPT_16_2P2: Esp32V2P2Inst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
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
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VZIPT_16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vzipt_16_2p2_p $qx, $qy, $qw",
			[(int_riscv_esp_vzipt_16_2p2 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_VZIPT_8_2P2: Esp32V2P2Inst<(outs QR:$qxr, QR:$qyr, QR:$qwr), (ins QR:$qx, QR:$qy, QR:$qw),
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
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 0;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VZIPT_8_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw),
				"!esp_vzipt_8_2p2_p $qx, $qy, $qw",
			[(int_riscv_esp_vzipt_8_2p2 timm:$qx, timm:$qy, timm:$qw)]>;

def ESP_ZERO_Q_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins),
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
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_ZERO_Q_2P2_P : PseudoEsp<(outs), (ins imm8:$qz),
				"!esp_zero_q_2p2_p $qz",
			[(int_riscv_esp_zero_q_2p2 timm:$qz)]>;

def ESP_ZERO_QACC_2P2: Esp32V2P2Inst<(outs), (ins),
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_ZERO_QACC_2P2_P : PseudoEsp<(outs), (ins),
				"!esp_zero_qacc_2p2_p",
			[(int_riscv_esp_zero_qacc_2p2)]>;

def ESP_ZERO_XACC_2P2: Esp32V2P2Inst<(outs), (ins),
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_ZERO_XACC_2P2_P : PseudoEsp<(outs), (ins),
				"!esp_zero_xacc_2p2_p",
			[(int_riscv_esp_zero_xacc_2p2)]>;

def ESP_FFT_AMS_S16_LD_INCP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2, sat:$sat),
                   "esp.fft.ams.s16.ld.incp\t $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<5> rs1;
  bits<1> sel2;
  bits<1> sat;
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
  let Inst{14} = sat{0};
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_AMS_S16_LD_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qw, GPRPIE:$rs1, select_2:$sel2, sat:$sat, imm8:$qu, imm8:$qz, imm8:$qv),
				"!esp_fft_ams_s16_ld_incp_2p2_p $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_ams_s16_ld_incp_2p2 timm:$qx, timm:$qy, timm:$qw, GPRPIE:$rs1, timm:$sel2, timm:$sat, timm:$qu, timm:$qz, timm:$qv))]>;

def ESP_FFT_AMS_S16_LD_INCP_UAUP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2, sat:$sat),
                   "esp.fft.ams.s16.ld.incp.uaup\t $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<5> rs1;
  bits<1> sel2;
  bits<1> sat;
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
  let Inst{14} = sat{0};
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
def ESP_FFT_AMS_S16_LD_INCP_UAUP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qw, GPRPIE:$rs1, select_2:$sel2, sat:$sat, imm8:$qu, imm8:$qz, imm8:$qv),
				"!esp_fft_ams_s16_ld_incp_uaup_2p2_p $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_ams_s16_ld_incp_uaup_2p2 timm:$qx, timm:$qy, timm:$qw, GPRPIE:$rs1, timm:$sel2, timm:$sat, timm:$qu, timm:$qz, timm:$qv))]>;

def ESP_FFT_AMS_S16_LD_R32_DECP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, QR:$qv, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, QR:$qw, GPRPIE:$rs1, select_2:$sel2, sat:$sat),
                   "esp.fft.ams.s16.ld.r32.decp\t $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<5> rs1;
  bits<1> sel2;
  bits<1> sat;
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
  let Inst{14} = sat{0};
  let Inst{13} = 1;
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
def ESP_FFT_AMS_S16_LD_R32_DECP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, imm8:$qw, GPRPIE:$rs1, select_2:$sel2, sat:$sat, imm8:$qu, imm8:$qz, imm8:$qv),
				"!esp_fft_ams_s16_ld_r32_decp_2p2_p $qu, $rs1, $qz, $qv, $qx, $qw, $qy, $sel2, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_ams_s16_ld_r32_decp_2p2 timm:$qx, timm:$qy, timm:$qw, GPRPIE:$rs1, timm:$sel2, timm:$sat, timm:$qu, timm:$qz, timm:$qv))]>;

def ESP_FFT_AMS_S16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r, GPRPIE:$rs2r), (ins QR:$qx, QR:$qy, QR:$qw, QR:$qu, GPRPIE:$rs1, GPRPIE:$rs2, select_2:$sel2, sat:$sat),
                   "esp.fft.ams.s16.st.incp\t $qu, $qz, $rs2, $rs1, $qx, $qw, $qy, $sel2, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qw;
  bits<3> qu;
  bits<5> rs1;
  bits<5> rs2;
  bits<1> sel2;
  bits<1> sat;
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
  let Inst{13} = sat{0};
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
def ESP_FFT_AMS_S16_ST_INCP_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qw, imm8:$qu, GPRPIE:$rs1, GPRPIE:$rs2, select_2:$sel2, sat:$sat, imm8:$qz),
				"!esp_fft_ams_s16_st_incp_2p2_p $qu, $qz, $rs2, $rs1, $qx, $qw, $qy, $sel2, $sat",
			[(int_riscv_esp_fft_ams_s16_st_incp_2p2 timm:$qx, timm:$qy, timm:$qw, timm:$qu, GPRPIE:$rs1, GPRPIE:$rs2, timm:$sel2, timm:$sat, timm:$qz)]>;

def ESP_FFT_BITREV_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r, QR:$qvr), (ins GPRPIE:$rs1, QR:$qv),
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
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_BITREV_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, imm8:$qv),
				"!esp_fft_bitrev_2p2_p $qv, $rs1",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_bitrev_2p2 GPRPIE:$rs1, timm:$qv))]>;

def ESP_FFT_CMUL_S16_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, select_8:$sel8),
                   "esp.fft.cmul.s16.ld.xp\t $qu, $rs1, $rs2, $qz, $qy, $qx, $sel8, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{25} = sat{0};
  let Inst{24} = sel8{2};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = sel8{1};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = sel8{0};
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_CMUL_S16_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, select_8:$sel8, imm8:$qz, imm8:$qu),
				"!esp_fft_cmul_s16_ld_xp_2p2_p $qu, $rs1, $rs2, $qz, $qy, $qx, $sel8, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_cmul_s16_ld_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$sel8, timm:$qz, timm:$qu))]>;

def ESP_FFT_CMUL_S16_ST_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qx, QR:$qy, QR:$qu, GPRPIE:$rs1, sat:$sat, select_4:$sel4, select_4:$upd4, select_8:$sel8),
                   "esp.fft.cmul.s16.st.xp\t $qy, $qx, $qu, $rs1, $rs2, $sel8, $upd4, $sel4, $sat", []> 
{
  bits<5> rs2;
  bits<3> qx;
  bits<3> qy;
  bits<3> qu;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{25} = sat{0};
  let Inst{24} = sel4{1};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = sel4{0};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14-13} = upd4{1-0};
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
def ESP_FFT_CMUL_S16_ST_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qx, imm8:$qy, imm8:$qu, GPRPIE:$rs1, sat:$sat, select_4:$sel4, select_4:$upd4, select_8:$sel8),
				"!esp_fft_cmul_s16_st_xp_2p2_p $qy, $qx, $qu, $rs1, $rs2, $sel8, $upd4, $sel4, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_cmul_s16_st_xp_2p2 GPRPIE:$rs2, timm:$qx, timm:$qy, timm:$qu, GPRPIE:$rs1, timm:$sat, timm:$sel4, timm:$upd4, timm:$sel8))]>;

def ESP_FFT_R2BF_S16_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qv), (ins QR:$qx, QR:$qy, select_2:$sel2, sat:$sat),
                   "esp.fft.r2bf.s16\t $qz, $qv, $qx, $qy, $sel2, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<1> sel2;
  bits<1> sat;
  bits<3> qz;
  bits<3> qv;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-29} = qx{2-0};
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = 0;
  let Inst{22-20} = qv{2-0};
  let Inst{19} = 0;
  let Inst{18} = sel2{0};
  let Inst{17} = sat{0};
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_FFT_R2BF_S16_2P2_P : PseudoEsp<(outs), (ins imm8:$qx, imm8:$qy, select_2:$sel2, sat:$sat, imm8:$qz, imm8:$qv),
				"!esp_fft_r2bf_s16_2p2_p $qz, $qv, $qx, $qy, $sel2, $sat",
			[(int_riscv_esp_fft_r2bf_s16_2p2 timm:$qx, timm:$qy, timm:$sel2, timm:$sat, timm:$qz, timm:$qv)]>;

def ESP_FFT_R2BF_S16_ST_INCP_2P2: Esp32V2P2Inst<(outs QR:$qz, GPRPIE:$rs1r), (ins QR:$qx, QR:$qy, GPRPIE:$rs1, sat:$sat, select_4:$sel4),
                   "esp.fft.r2bf.s16.st.incp\t $qz, $qx, $qy, $rs1, $sel4, $sat", []> 
{
  bits<3> qx;
  bits<3> qy;
  bits<5> rs1;
  bits<1> sat;
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
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = sat{0};
  let Inst{11-10} = sel4{1-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_R2BF_S16_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qx, imm8:$qy, GPRPIE:$rs1, sat:$sat, select_4:$sel4, imm8:$qz),
				"!esp_fft_r2bf_s16_st_incp_2p2_p $qz, $qx, $qy, $rs1, $sel4, $sat",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_r2bf_s16_st_incp_2p2 timm:$qx, timm:$qy, GPRPIE:$rs1, timm:$sat, timm:$sel4, timm:$qz))]>;

def ESP_FFT_VST_R32_DECP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, select_2:$sel2),
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
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_FFT_VST_R32_DECP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, select_2:$sel2),
				"!esp_fft_vst_r32_decp_2p2_p $qu, $rs1, $sel2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_fft_vst_r32_decp_2p2 timm:$qu, GPRPIE:$rs1, timm:$sel2))]>;

def ESP_LD_128_USAR_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
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
  
  let Inst{31-26} = off25616{7-2};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off25616{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_128_USAR_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616, imm8:$qu),
				"!esp_ld_128_usar_ip_2p2_p $qu, $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_128_usar_ip_2p2 GPRPIE:$rs1, timm:$off25616, timm:$qu))]>;

def ESP_LD_128_USAR_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
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
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_128_USAR_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_ld_128_usar_xp_2p2_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_128_usar_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_LD_XACC_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.ld.xacc.ip\t $rs1, $off2568", []> 
{
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = off2568{7-5};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off2568{4-2};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-11} = off2568{1-0};
  let Inst{10} = 0;
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_XACC_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_ld_xacc_ip_2p2_p $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_xacc_ip_2p2 GPRPIE:$rs1, timm:$off2568))]>;

def ESP_LDQA_S16_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ldqa.s16.128.ip\t $rs1, $off25616", []> 
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
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_S16_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_s16_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s16_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDQA_S16_128_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ldqa.s16.128.xp\t $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
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
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_S16_128_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_s16_128_xp_2p2_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s16_128_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1))]>;

def ESP_LDQA_S8_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
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
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_S8_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_s8_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s8_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDQA_S8_128_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  let Inst{30} = 1;
  let Inst{29} = 0;
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
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_S8_128_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_s8_128_xp_2p2_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_s8_128_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1))]>;

def ESP_LDQA_U16_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.ldqa.u16.128.ip\t $rs1, $off25616", []> 
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
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_U16_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_u16_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u16_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDQA_U16_128_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
                   "esp.ldqa.u16.128.xp\t $rs1, $rs2", []> 
{
  bits<5> rs2;
  bits<5> rs1;
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
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_U16_128_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_u16_128_xp_2p2_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u16_128_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1))]>;

def ESP_LDQA_U8_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
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
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_U8_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ldqa_u8_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u8_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDQA_U8_128_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LDQA_U8_128_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
				"!esp_ldqa_u8_128_xp_2p2_p $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ldqa_u8_128_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1))]>;

def ESP_VLDBC_16_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_2:$offset_256_2),
                   "esp.vldbc.16.ip\t $qu, $rs1, $offset_256_2", []> 
{
  bits<5> rs1;
  bits<8> offset_256_2;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-26} = offset_256_2{7-2};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = offset_256_2{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_16_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_2:$offset_256_2, imm8:$qu),
				"!esp_vldbc_16_ip_2p2_p $qu, $rs1, $offset_256_2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_16_ip_2p2 GPRPIE:$rs1, timm:$offset_256_2, timm:$qu))]>;

def ESP_VLDBC_16_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_16_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vldbc_16_xp_2p2_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_16_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLDBC_32_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_4:$off2564),
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
  
  let Inst{31-26} = off2564{7-2};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = off2564{1-0};
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_32_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_4:$off2564, imm8:$qu),
				"!esp_vldbc_32_ip_2p2_p $qu, $rs1, $off2564",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_32_ip_2p2 GPRPIE:$rs1, timm:$off2564, timm:$qu))]>;

def ESP_VLDBC_32_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_32_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vldbc_32_xp_2p2_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_32_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLDBC_8_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_1:$offset_256_1),
                   "esp.vldbc.8.ip\t $qu, $rs1, $offset_256_1", []> 
{
  bits<5> rs1;
  bits<8> offset_256_1;
  bits<3> qu;
  bits<5> rs1r;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31-26} = offset_256_1{7-2};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = offset_256_1{1-0};
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_8_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_1:$offset_256_1, imm8:$qu),
				"!esp_vldbc_8_ip_2p2_p $qu, $rs1, $offset_256_1",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_8_ip_2p2 GPRPIE:$rs1, timm:$offset_256_1, timm:$qu))]>;

def ESP_VLDBC_8_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 1;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDBC_8_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vldbc_8_xp_2p2_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldbc_8_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLDEXT_S16_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
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
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = off1616{3};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off1616{2-0};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_S16_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_s16_ip_2p2_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s16_ip_2p2 GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_S16_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
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
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_S16_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_s16_xp_2p2_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s16_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_S8_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
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
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = off1616{3};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off1616{2-0};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_S8_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_s8_ip_2p2_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s8_ip_2p2 GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_S8_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_S8_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_s8_xp_2p2_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_s8_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U16_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
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
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = off1616{3};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off1616{2-0};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_U16_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_u16_ip_2p2_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u16_ip_2p2 GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U16_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
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
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_U16_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_u16_xp_2p2_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u16_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U8_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616),
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
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = off1616{3};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off1616{2-0};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_U8_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_16_16:$off1616, imm8:$qu, imm8:$qz),
				"!esp_vldext_u8_ip_2p2_p $qu, $qz, $rs1, $off1616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u8_ip_2p2 GPRPIE:$rs1, timm:$off1616, timm:$qu, timm:$qz))]>;

def ESP_VLDEXT_U8_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  let Inst{30} = 1;
  let Inst{29} = 0;
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDEXT_U8_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldext_u8_xp_2p2_p $qu, $qz, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldext_u8_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_VLDHBC_16_INCP_2P2: Esp32V2P2Inst<(outs QR:$qu, QR:$qz, GPRPIE:$rs1r), (ins GPRPIE:$rs1),
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
  
  let Inst{31} = 1;
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
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLDHBC_16_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, imm8:$qu, imm8:$qz),
				"!esp_vldhbc_16_incp_2p2_p $qu, $qz, $rs1",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vldhbc_16_incp_2p2 GPRPIE:$rs1, timm:$qu, timm:$qz))]>;

def ESP_LD_QACC_H_H_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
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
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_QACC_H_H_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_h_h_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_h_h_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LD_QACC_H_L_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
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
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_QACC_H_L_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_h_l_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_h_l_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LD_QACC_L_H_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
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
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_QACC_L_H_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_l_h_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_l_h_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LD_QACC_L_L_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
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
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_QACC_L_L_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_qacc_l_l_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_qacc_l_l_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LD_UA_STATE_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
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
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27-26} = off25616{7-6};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{5-3};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = off25616{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_LD_UA_STATE_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_ld_ua_state_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_ld_ua_state_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_LDXQ_32_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins GPRPIE:$rs1, QR:$qw, select_4:$sel4, select_8:$sel8),
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
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22-21} = sel4{1-0};
  let Inst{20} = sel8{2};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = sel8{1-0};
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_LDXQ_32_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qw, select_4:$sel4, select_8:$sel8, imm8:$qu),
				"!esp_ldxq_32_2p2_p $qu, $qw, $rs1, $sel4, $sel8",
			[(int_riscv_esp_ldxq_32_2p2 GPRPIE:$rs1, timm:$qw, timm:$sel4, timm:$sel8, timm:$qu)]>;

def ESP_ST_QACC_H_H_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.h.h.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_QACC_H_H_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_h_h_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_h_h_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_ST_QACC_H_L_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.h.l.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_QACC_H_L_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_h_l_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_h_l_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_ST_QACC_L_H_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.l.h.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_QACC_L_H_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_l_h_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_l_h_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_ST_QACC_L_L_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.qacc.l.l.128.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = off25616{7};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{6-4};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-9} = off25616{3-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_QACC_L_L_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_qacc_l_l_128_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_qacc_l_l_128_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_ST_UA_STATE_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                   "esp.st.ua.state.ip\t $rs1, $off25616", []> 
{
  bits<5> rs1;
  bits<8> off25616;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27-26} = off25616{7-6};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off25616{5-3};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = off25616{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_UA_STATE_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_st_ua_state_ip_2p2_p $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_ua_state_ip_2p2 GPRPIE:$rs1, timm:$off25616))]>;

def ESP_STXQ_32_2P2: Esp32V2P2Inst<(outs), (ins GPRPIE:$rs1, QR:$qw, QR:$qu, select_4:$sel4, select_8:$sel8),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22-21} = sel4{1-0};
  let Inst{20} = sel8{2};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9-8} = sel8{1-0};
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_STXQ_32_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, imm8:$qw, imm8:$qu, select_4:$sel4, select_8:$sel8),
				"!esp_stxq_32_2p2_p $qu, $qw, $rs1, $sel4, $sel8",
			[(int_riscv_esp_stxq_32_2p2 GPRPIE:$rs1, timm:$qw, timm:$qu, timm:$sel4, timm:$sel8)]>;

def ESP_VLD_128_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
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
  let Inst{30-26} = off25616{7-3};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = off25616{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616, imm8:$qu),
				"!esp_vld_128_ip_2p2_p $qu, $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_128_ip_2p2 GPRPIE:$rs1, timm:$off25616, timm:$qu))]>;

def ESP_VLD_128_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
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
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_128_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vld_128_xp_2p2_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_128_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLD_H_64_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
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
  let Inst{29-26} = off2568{7-4};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = off2568{3};
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = off2568{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_H_64_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568, imm8:$qu),
				"!esp_vld_h_64_ip_2p2_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_h_64_ip_2p2 GPRPIE:$rs1, timm:$off2568, timm:$qu))]>;

def ESP_VLD_H_64_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
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
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_H_64_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vld_h_64_xp_2p2_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_h_64_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VLD_L_64_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
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
  let Inst{29-26} = off2568{7-4};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = off2568{3};
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = off2568{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_L_64_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568, imm8:$qu),
				"!esp_vld_l_64_ip_2p2_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_l_64_ip_2p2 GPRPIE:$rs1, timm:$off2568, timm:$qu))]>;

def ESP_VLD_L_64_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
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
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VLD_L_64_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, GPRPIE:$rs1, imm8:$qu),
				"!esp_vld_l_64_xp_2p2_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vld_l_64_xp_2p2 GPRPIE:$rs2, GPRPIE:$rs1, timm:$qu))]>;

def ESP_VST_128_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, offset_256_16:$off25616),
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
  let Inst{30-26} = off25616{7-3};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = 0;
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = off25616{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_128_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, offset_256_16:$off25616),
				"!esp_vst_128_ip_2p2_p $qu, $rs1, $off25616",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_128_ip_2p2 timm:$qu, GPRPIE:$rs1, timm:$off25616))]>;

def ESP_VST_128_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qu, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
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
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_128_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qu, GPRPIE:$rs1),
				"!esp_vst_128_xp_2p2_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_128_xp_2p2 GPRPIE:$rs2, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VST_H_64_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
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
  let Inst{29-26} = off2568{7-4};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = off2568{3};
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = off2568{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_H_64_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_vst_h_64_ip_2p2_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_h_64_ip_2p2 timm:$qu, GPRPIE:$rs1, timm:$off2568))]>;

def ESP_VST_H_64_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qu, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
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
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_H_64_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qu, GPRPIE:$rs1),
				"!esp_vst_h_64_xp_2p2_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_h_64_xp_2p2 GPRPIE:$rs2, timm:$qu, GPRPIE:$rs1))]>;

def ESP_VST_L_64_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
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
  let Inst{29-26} = off2568{7-4};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = off2568{3};
  let Inst{22} = 1;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9-7} = off2568{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_L_64_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qu, GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_vst_l_64_ip_2p2_p $qu, $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_l_64_ip_2p2 timm:$qu, GPRPIE:$rs1, timm:$off2568))]>;

def ESP_VST_L_64_XP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, QR:$qu, GPRPIE:$rs1),
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
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = 0;
  let Inst{25} = 1;
  let Inst{24} = 0;
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
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
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_VST_L_64_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qu, GPRPIE:$rs1),
				"!esp_vst_l_64_xp_2p2_p $qu, $rs1, $rs2",
			[(set GPRPIE:$rs1r, (int_riscv_esp_vst_l_64_xp_2p2 GPRPIE:$rs2, timm:$qu, GPRPIE:$rs1))]>;

def ESP_SLCI_2Q_2P2: Esp32V2P2Inst<(outs QR:$qyr, QR:$qwr), (ins QR:$qy, QR:$qw, select_16:$sel16),
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
  let Inst{30-29} = sel16{3-2};
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10-9} = sel16{1-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SLCI_2Q_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, select_16:$sel16),
				"!esp_slci_2q_2p2_p $qy, $qw, $sel16",
			[(int_riscv_esp_slci_2q_2p2 timm:$qy, timm:$qw, timm:$sel16)]>;

def ESP_SLCXXP_2Q_2P2: Esp32V2P2Inst<(outs QR:$qyr, QR:$qwr), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy, QR:$qw),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SLCXXP_2Q_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy, imm8:$qw),
				"!esp_slcxxp_2q_2p2_p $qy, $qw, $rs1, $rs2",
			[(int_riscv_esp_slcxxp_2q_2p2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy, timm:$qw)]>;

def ESP_SRC_Q_2P2: Esp32V2P2Inst<(outs QR:$qz), (ins QR:$qy, QR:$qw),
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
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRC_Q_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qz),
				"!esp_src_q_2p2_p $qz, $qw, $qy",
			[(int_riscv_esp_src_q_2p2 timm:$qy, timm:$qw, timm:$qz)]>;

def ESP_SRC_Q_LD_IP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r, QR:$qwr), (ins QR:$qy, GPRPIE:$rs1, QR:$qw, offset_256_16:$off25616),
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
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = off25616{0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_SRC_Q_LD_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qy, GPRPIE:$rs1, imm8:$qw, offset_256_16:$off25616, imm8:$qu),
				"!esp_src_q_ld_ip_2p2_p $qu, $rs1, $off25616, $qw, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_src_q_ld_ip_2p2 timm:$qy, GPRPIE:$rs1, timm:$qw, timm:$off25616, timm:$qu))]>;

def ESP_SRC_Q_LD_XP_2P2: Esp32V2P2Inst<(outs QR:$qu, GPRPIE:$rs1r, QR:$qwr), (ins GPRPIE:$rs2, QR:$qy, GPRPIE:$rs1, QR:$qw),
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
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = rs2{4};
  let Inst{22-20} = rs2{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_SRC_Q_LD_XP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs2, imm8:$qy, GPRPIE:$rs1, imm8:$qw, imm8:$qu),
				"!esp_src_q_ld_xp_2p2_p $qu, $rs1, $rs2, $qw, $qy",
			[(set GPRPIE:$rs1r, (int_riscv_esp_src_q_ld_xp_2p2 GPRPIE:$rs2, timm:$qy, GPRPIE:$rs1, timm:$qw, timm:$qu))]>;

def ESP_SRC_Q_QUP_2P2: Esp32V2P2Inst<(outs QR:$qz, QR:$qwr), (ins QR:$qy, QR:$qw),
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
  let Inst{21} = 0;
  let Inst{20} = 0;
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 0;
  let Inst{9-7} = qz{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRC_Q_QUP_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, imm8:$qz),
				"!esp_src_q_qup_2p2_p $qz, $qw, $qy",
			[(int_riscv_esp_src_q_qup_2p2 timm:$qy, timm:$qw, timm:$qz)]>;

def ESP_SRCI_2Q_2P2: Esp32V2P2Inst<(outs QR:$qyr, QR:$qwr), (ins QR:$qy, QR:$qw, select_16:$sel16),
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
  
  let Inst{31} = 1;
  let Inst{30-29} = sel16{3-2};
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10-9} = sel16{1-0};
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCI_2Q_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, select_16:$sel16),
				"!esp_srci_2q_2p2_p $qy, $qw, $sel16",
			[(int_riscv_esp_srci_2q_2p2 timm:$qy, timm:$qw, timm:$sel16)]>;

def ESP_SRCMB_S16_Q_QACC_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qw, sat:$sat, rm:$rm),
                   "esp.srcmb.s16.q.qacc\t $qu, $qw, $sat, $rm", []> 
{
  bits<3> qw;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 1;
  let Inst{26} = sat{0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22-20} = rm{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_S16_Q_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_srcmb_s16_q_qacc_2p2_p $qu, $qw, $sat, $rm",
			[(int_riscv_esp_srcmb_s16_q_qacc_2p2 timm:$qw, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_SRCMB_S16_QACC_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.srcmb.s16.qacc\t $qu, $rs1, $sat, $rm", []> 
{
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = sat{0};
  let Inst{28-26} = rm{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_S16_QACC_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_srcmb_s16_qacc_2p2_p $qu, $rs1, $sat, $rm",
			[(int_riscv_esp_srcmb_s16_qacc_2p2 GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_SRCMB_S8_Q_QACC_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qw, sat:$sat, rm:$rm),
                   "esp.srcmb.s8.q.qacc\t $qu, $qw, $sat, $rm", []> 
{
  bits<3> qw;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 1;
  let Inst{26} = sat{0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22-20} = rm{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_S8_Q_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_srcmb_s8_q_qacc_2p2_p $qu, $qw, $sat, $rm",
			[(int_riscv_esp_srcmb_s8_q_qacc_2p2 timm:$qw, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_SRCMB_S8_QACC_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.srcmb.s8.qacc\t $qu, $rs1, $sat, $rm", []> 
{
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 1;
  let Inst{29} = sat{0};
  let Inst{28-26} = rm{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_S8_QACC_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_srcmb_s8_qacc_2p2_p $qu, $rs1, $sat, $rm",
			[(int_riscv_esp_srcmb_s8_qacc_2p2 GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_SRCMB_U16_Q_QACC_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qw, sat:$sat, rm:$rm),
                   "esp.srcmb.u16.q.qacc\t $qu, $qw, $sat, $rm", []> 
{
  bits<3> qw;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = 0;
  let Inst{26} = sat{0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22-20} = rm{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_U16_Q_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_srcmb_u16_q_qacc_2p2_p $qu, $qw, $sat, $rm",
			[(int_riscv_esp_srcmb_u16_q_qacc_2p2 timm:$qw, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_SRCMB_U16_QACC_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.srcmb.u16.qacc\t $qu, $rs1, $sat, $rm", []> 
{
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = sat{0};
  let Inst{28-26} = rm{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_U16_QACC_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_srcmb_u16_qacc_2p2_p $qu, $rs1, $sat, $rm",
			[(int_riscv_esp_srcmb_u16_qacc_2p2 GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_SRCMB_U8_Q_QACC_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qw, sat:$sat, rm:$rm),
                   "esp.srcmb.u8.q.qacc\t $qu, $qw, $sat, $rm", []> 
{
  bits<3> qw;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = 0;
  let Inst{26} = sat{0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22-20} = rm{2-0};
  let Inst{19} = qw{2};
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 1;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_U8_Q_QACC_2P2_P : PseudoEsp<(outs), (ins imm8:$qw, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_srcmb_u8_q_qacc_2p2_p $qu, $qw, $sat, $rm",
			[(int_riscv_esp_srcmb_u8_q_qacc_2p2 timm:$qw, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_SRCMB_U8_QACC_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.srcmb.u8.qacc\t $qu, $rs1, $sat, $rm", []> 
{
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = sat{0};
  let Inst{28-26} = rm{2-0};
  let Inst{25} = 1;
  let Inst{24} = 1;
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = 1;
  let Inst{20} = 0;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCMB_U8_QACC_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_srcmb_u8_qacc_2p2_p $qu, $rs1, $sat, $rm",
			[(int_riscv_esp_srcmb_u8_qacc_2p2 GPRPIE:$rs1, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_SRCQ_128_ST_INCP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins QR:$qy, QR:$qw, GPRPIE:$rs1),
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
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
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
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_SRCQ_128_ST_INCP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins imm8:$qy, imm8:$qw, GPRPIE:$rs1),
				"!esp_srcq_128_st_incp_2p2_p $qw, $qy, $rs1",
			[(set GPRPIE:$rs1r, (int_riscv_esp_srcq_128_st_incp_2p2 timm:$qy, timm:$qw, GPRPIE:$rs1))]>;

def ESP_SRCXXP_2Q_2P2: Esp32V2P2Inst<(outs QR:$qyr, QR:$qwr), (ins GPRPIE:$rs1, GPRPIE:$rs2, QR:$qy, QR:$qw),
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
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = 1;
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRCXXP_2Q_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, GPRPIE:$rs2, imm8:$qy, imm8:$qw),
				"!esp_srcxxp_2q_2p2_p $qy, $qw, $rs1, $rs2",
			[(int_riscv_esp_srcxxp_2q_2p2 GPRPIE:$rs1, GPRPIE:$rs2, timm:$qy, timm:$qw)]>;

def ESP_SRS_S_XACC_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.srs.s.xacc\t $rd, $rs1, $sat, $rm", []> 
{
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27} = sat{0};
  let Inst{26} = rm{2};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-21} = rm{1-0};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRS_S_XACC_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, sat:$sat, rm:$rm, GPRPIE:$rd),
				"!esp_srs_s_xacc_2p2_p $rd, $rs1, $sat, $rm",
			[(int_riscv_esp_srs_s_xacc_2p2 GPRPIE:$rs1, timm:$sat, timm:$rm, GPRPIE:$rd)]>;

def ESP_SRS_U_XACC_2P2: Esp32V2P2Inst<(outs GPRPIE:$rd), (ins GPRPIE:$rs1, sat:$sat, rm:$rm),
                   "esp.srs.u.xacc\t $rd, $rs1, $sat, $rm", []> 
{
  bits<5> rs1;
  bits<1> sat;
  bits<3> rm;
  bits<5> rd;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27} = sat{0};
  let Inst{26} = rm{2};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23} = 1;
  let Inst{22-21} = rm{1-0};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11} = 0;
  let Inst{10} = rd{4};
  let Inst{9-7} = rd{2-0};
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_SRS_U_XACC_2P2_P : PseudoEsp<(outs), (ins GPRPIE:$rs1, sat:$sat, rm:$rm, GPRPIE:$rd),
				"!esp_srs_u_xacc_2p2_p $rd, $rs1, $sat, $rm",
			[(int_riscv_esp_srs_u_xacc_2p2 GPRPIE:$rs1, timm:$sat, timm:$rm, GPRPIE:$rd)]>;

def ESP_VSL_S32_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qy, sat:$sat),
                   "esp.vsl.s32\t $qu, $qy, $sat", []> 
{
  bits<3> qy;
  bits<1> sat;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = sat{0};
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
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSL_S32_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, sat:$sat, imm8:$qu),
				"!esp_vsl_s32_2p2_p $qu, $qy, $sat",
			[(int_riscv_esp_vsl_s32_2p2 timm:$qy, timm:$sat, timm:$qu)]>;

def ESP_VSL_U32_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qy, sat:$sat),
                   "esp.vsl.u32\t $qu, $qy, $sat", []> 
{
  bits<3> qy;
  bits<1> sat;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = sat{0};
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
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSL_U32_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, sat:$sat, imm8:$qu),
				"!esp_vsl_u32_2p2_p $qu, $qy, $sat",
			[(int_riscv_esp_vsl_u32_2p2 timm:$qy, timm:$sat, timm:$qu)]>;

def ESP_VSLD_16_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw, sat:$sat, rm:$rm),
                   "esp.vsld.16\t $qu, $qy, $qw, $sat, $rm", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 1;
  let Inst{21} = sat{0};
  let Inst{20} = rm{2};
  let Inst{19} = qw{2};
  let Inst{18-17} = rm{1-0};
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSLD_16_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_vsld_16_2p2_p $qu, $qy, $qw, $sat, $rm",
			[(int_riscv_esp_vsld_16_2p2 timm:$qy, timm:$qw, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_VSLD_32_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw, sat:$sat, rm:$rm),
                   "esp.vsld.32\t $qu, $qy, $qw, $sat, $rm", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = sat{0};
  let Inst{21-20} = rm{2-1};
  let Inst{19} = qw{2};
  let Inst{18} = rm{0};
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSLD_32_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_vsld_32_2p2_p $qu, $qy, $qw, $sat, $rm",
			[(int_riscv_esp_vsld_32_2p2 timm:$qy, timm:$qw, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_VSLD_8_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw, sat:$sat, rm:$rm),
                   "esp.vsld.8\t $qu, $qy, $qw, $sat, $rm", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 0;
  let Inst{22} = 0;
  let Inst{21} = sat{0};
  let Inst{20} = rm{2};
  let Inst{19} = qw{2};
  let Inst{18-17} = rm{1-0};
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSLD_8_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_vsld_8_2p2_p $qu, $qy, $qw, $sat, $rm",
			[(int_riscv_esp_vsld_8_2p2 timm:$qy, timm:$qw, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_VSR_S32_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qy, rm:$rm),
                   "esp.vsr.s32\t $qu, $qy, $rm", []> 
{
  bits<3> qy;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 1;
  let Inst{24} = rm{2};
  let Inst{23} = 0;
  let Inst{22-21} = rm{1-0};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSR_S32_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, rm:$rm, imm8:$qu),
				"!esp_vsr_s32_2p2_p $qu, $qy, $rm",
			[(int_riscv_esp_vsr_s32_2p2 timm:$qy, timm:$rm, timm:$qu)]>;

def ESP_VSR_U32_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qy, rm:$rm),
                   "esp.vsr.u32\t $qu, $qy, $rm", []> 
{
  bits<3> qy;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25} = 0;
  let Inst{24} = rm{2};
  let Inst{23} = 0;
  let Inst{22-21} = rm{1-0};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = 1;
  let Inst{17} = 0;
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 0;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSR_U32_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, rm:$rm, imm8:$qu),
				"!esp_vsr_u32_2p2_p $qu, $qy, $rm",
			[(int_riscv_esp_vsr_u32_2p2 timm:$qy, timm:$rm, timm:$qu)]>;

def ESP_VSRD_16_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw, sat:$sat, rm:$rm),
                   "esp.vsrd.16\t $qu, $qy, $qw, $sat, $rm", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 1;
  let Inst{21} = sat{0};
  let Inst{20} = rm{2};
  let Inst{19} = qw{2};
  let Inst{18-17} = rm{1-0};
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSRD_16_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_vsrd_16_2p2_p $qu, $qy, $qw, $sat, $rm",
			[(int_riscv_esp_vsrd_16_2p2 timm:$qy, timm:$qw, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_VSRD_32_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw, sat:$sat, rm:$rm),
                   "esp.vsrd.32\t $qu, $qy, $qw, $sat, $rm", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = sat{0};
  let Inst{21-20} = rm{2-1};
  let Inst{19} = qw{2};
  let Inst{18} = rm{0};
  let Inst{17} = 0;
  let Inst{16} = 1;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSRD_32_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_vsrd_32_2p2_p $qu, $qy, $qw, $sat, $rm",
			[(int_riscv_esp_vsrd_32_2p2 timm:$qy, timm:$qw, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_VSRD_8_2P2: Esp32V2P2Inst<(outs QR:$qu), (ins QR:$qy, QR:$qw, sat:$sat, rm:$rm),
                   "esp.vsrd.8\t $qu, $qy, $qw, $sat, $rm", []> 
{
  bits<3> qy;
  bits<3> qw;
  bits<1> sat;
  bits<3> rm;
  bits<3> qu;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29} = 0;
  let Inst{28-26} = qy{2-0};
  let Inst{25-24} = qw{1-0};
  let Inst{23} = 1;
  let Inst{22} = 0;
  let Inst{21} = sat{0};
  let Inst{20} = rm{2};
  let Inst{19} = qw{2};
  let Inst{18-17} = rm{1-0};
  let Inst{16} = 0;
  let Inst{15} = 0;
  let Inst{14} = 1;
  let Inst{13} = 1;
  let Inst{12-10} = qu{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 0;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1 in 
def ESP_VSRD_8_2P2_P : PseudoEsp<(outs), (ins imm8:$qy, imm8:$qw, sat:$sat, rm:$rm, imm8:$qu),
				"!esp_vsrd_8_2p2_p $qu, $qy, $qw, $sat, $rm",
			[(int_riscv_esp_vsrd_8_2p2 timm:$qy, timm:$qw, timm:$sat, timm:$rm, timm:$qu)]>;

def ESP_ST_S_XACC_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
                   "esp.st.s.xacc.ip\t $rs1, $off2568", []> 
{
  bits<5> rs1;
  bits<8> off2568;
  bits<5> rs1r;
  let mayStore = 1;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Constraints = "$rs1r = $rs1";
  
  let Inst{31} = 0;
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 1;
  let Inst{27-26} = off2568{7-6};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off2568{5-3};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = off2568{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_S_XACC_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_st_s_xacc_ip_2p2_p $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_s_xacc_ip_2p2 GPRPIE:$rs1, timm:$off2568))]>;

def ESP_ST_U_XACC_IP_2P2: Esp32V2P2Inst<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
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
  let Inst{30} = 0;
  let Inst{29} = 0;
  let Inst{28} = 0;
  let Inst{27-26} = off2568{7-6};
  let Inst{25} = 0;
  let Inst{24} = 0;
  let Inst{23-21} = off2568{5-3};
  let Inst{20} = 1;
  let Inst{19} = 0;
  let Inst{18} = rs1{4};
  let Inst{17-15} = rs1{2-0};
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12-10} = off2568{2-0};
  let Inst{9} = 0;
  let Inst{8} = 1;
  let Inst{7} = 1;
  let Inst{6} = 0;
  let Inst{5} = 0;
  let Inst{4} = 1;
  let Inst{3} = 1;
  let Inst{2} = 1;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ST_U_XACC_IP_2P2_P : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_8:$off2568),
				"!esp_st_u_xacc_ip_2p2_p $rs1, $off2568",
			[(set GPRPIE:$rs1r, (int_riscv_esp_st_u_xacc_ip_2p2 GPRPIE:$rs1, timm:$off2568))]>;



