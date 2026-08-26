; Shared .m VLD/VST.[L/H].64.IP select 2.1 vs pure 2.2 opcodes.
; RUN: llc -O2 -mtriple=riscv32-esp-unknown-elf -mcpu=esp32p4 -mattr=+espv-lowering \
; RUN:   -stop-after=finalize-isel -verify-machineinstrs %s -o - \
; RUN:   | FileCheck %s --check-prefix=ESPV22
; RUN: llc -O2 -mtriple=riscv32-esp-unknown-elf -mcpu=esp32p4eco4 -mattr=+espv-lowering \
; RUN:   -stop-after=finalize-isel -verify-machineinstrs %s -o - \
; RUN:   | FileCheck %s --check-prefix=ESPV21

; ESPV22-LABEL: name: copy8_l
; ESPV22: ESP_VLD_L_64_IP_2P2
; ESPV22: ESP_VST_L_64_IP_2P2
; ESPV21-LABEL: name: copy8_l
; ESPV21: ESP_VLD_L_64_IP
; ESPV21-NOT: ESP_VLD_L_64_IP_2P2
; ESPV21: ESP_VST_L_64_IP
; ESPV21-NOT: ESP_VST_L_64_IP_2P2

define ptr @copy8_l(ptr %dst, ptr %src) {
entry:
  %load = call { <8 x i8>, ptr } @llvm.riscv.esp.vld.l.64.ip(ptr %src, i32 8)
  %value = extractvalue { <8 x i8>, ptr } %load, 0
  %next.dst = call ptr @llvm.riscv.esp.vst.l.64.ip(<8 x i8> %value, ptr %dst, i32 8)
  ret ptr %next.dst
}

; ESPV22-LABEL: name: copy8_h
; ESPV22: ESP_VLD_H_64_IP_2P2
; ESPV22: ESP_VST_H_64_IP_2P2
; ESPV21-LABEL: name: copy8_h
; ESPV21: ESP_VLD_H_64_IP
; ESPV21-NOT: ESP_VLD_H_64_IP_2P2

define ptr @copy8_h(ptr %dst, ptr %src) {
entry:
  %load = call { <8 x i8>, ptr } @llvm.riscv.esp.vld.h.64.ip(ptr %src, i32 8)
  %value = extractvalue { <8 x i8>, ptr } %load, 0
  %next.dst = call ptr @llvm.riscv.esp.vst.h.64.ip(<8 x i8> %value, ptr %dst, i32 8)
  ret ptr %next.dst
}

declare { <8 x i8>, ptr } @llvm.riscv.esp.vld.l.64.ip(ptr, i32)
declare ptr @llvm.riscv.esp.vst.l.64.ip(<8 x i8>, ptr, i32)
declare { <8 x i8>, ptr } @llvm.riscv.esp.vld.h.64.ip(ptr, i32)
declare ptr @llvm.riscv.esp.vst.h.64.ip(<8 x i8>, ptr, i32)
