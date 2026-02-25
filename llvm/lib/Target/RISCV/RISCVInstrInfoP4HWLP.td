//===- RISCVInstrInfoP4HWLP.td - RISCV Target Description -*- tablegen -*--===//
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

def ESP_LP_SETUPI: Esp32P4Inst<(outs), (ins uimm1:$id, uimm12:$count, uimm10_step4:$offset),
                   "esp.lp.setupi\t $id, $count, $offset", []> 
{
  bits<1> id;
  bits<13> count;
  bits<10> offset;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-20} = count{11-0};
  let Inst{19-15} = offset{5-1};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-8} = offset{9-6};
  let Inst{7} = id;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 0;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LP_SETUP: Esp32P4Inst<(outs), (ins uimm1:$id, GPR:$rs1, uimm13_step4:$offset),
                   "esp.lp.setup\t $id, $rs1, $offset", []> 
{
  bits<1> id;
  bits<5> rs1;
  bits<13> offset;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-20} = offset{12-1};
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 1;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11-8} = 0xb0000;
  let Inst{7} = id;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 0;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LP_STARTI: Esp32P4Inst<(outs), (ins uimm1:$id, uimm13_step4:$offset),
                   "esp.lp.starti\t $id, $offset", []> 
{
  bits<1> id;
  bits<5> rs1;
  bits<13> offset;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-20} = offset{12-1};
  let Inst{19-15} = 0xb00000;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11-8} = 0xb0000;
  let Inst{7} = id;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 0;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LP_ENDI: Esp32P4Inst<(outs), (ins uimm1:$id, uimm13_step4:$offset),
                   "esp.lp.endi\t $id, $offset", []> 
{
  bits<1> id;
  bits<5> rs1;
  bits<13> offset;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-20} = offset{12-1};
  let Inst{19-15} = 0xb00000;
  let Inst{14} = 0;
  let Inst{13} = 0;
  let Inst{12} = 1;
  let Inst{11-8} = 0xb0000;
  let Inst{7} = id;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 0;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LP_COUNTI: Esp32P4Inst<(outs), (ins uimm1:$id, uimm12:$count),
                   "esp.lp.counti\t $id, $count", []> 
{
  bits<1> id;
  bits<5> rs1;
  bits<12> count;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-20} = count{11-0};
  let Inst{19-15} = 0xb00000;
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 1;
  let Inst{11-8} = 0xb0000;
  let Inst{7} = id;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 0;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}

def ESP_LP_COUNT: Esp32P4Inst<(outs), (ins uimm1:$id, GPR:$rs1),
                   "esp.lp.count\t $id, $rs1", []> 
{
  bits<1> id;
  bits<5> rs1;
  bits<13> offset;
  let mayStore = 0;
  let mayLoad = 0;
  let hasSideEffects = 1;
  
  let Inst{31-20} = 0xb000000000000;
  let Inst{19-15} = rs1{4-0};
  let Inst{14} = 0;
  let Inst{13} = 1;
  let Inst{12} = 0;
  let Inst{11-8} = 0xb0000;
  let Inst{7} = id;
  let Inst{6} = 0;
  let Inst{5} = 1;
  let Inst{4} = 0;
  let Inst{3} = 1;
  let Inst{2} = 0;
  let Inst{1} = 1;
  let Inst{0} = 1;
}