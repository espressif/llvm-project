; ESPV lowering stays opt-in (+espv-lowering); -mcpu only selects the ISA level.
; RUN: llc -O2 -mtriple=riscv32-esp-unknown-elf -mcpu=esp32p4 -mattr=+espv-lowering \
; RUN:   -stop-after=finalize-isel -verify-machineinstrs %s -o - \
; RUN:   | FileCheck %s --check-prefix=ESPV22
; RUN: llc -O2 -mtriple=riscv32-esp-unknown-elf -mcpu=esp32p4eco4 -mattr=+espv-lowering \
; RUN:   -stop-after=finalize-isel -verify-machineinstrs %s -o - \
; RUN:   | FileCheck %s --check-prefix=ESPV21

; ESPV22-LABEL: name: copy16
; ESPV22: ESP_VLD_128_IP_2P2
; ESPV22: ESP_VST_128_IP_2P2
; ESPV21-LABEL: name: copy16
; ESPV21: ESP_VLD_128_IP
; ESPV21-NOT: ESP_VLD_128_IP_2P2
; ESPV21: ESP_VST_128_IP
; ESPV21-NOT: ESP_VST_128_IP_2P2

define ptr @copy16(ptr %dst, ptr %src) {
entry:
  %load = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip.m(ptr %src,
                                                               i32 16)
  %value = extractvalue { <16 x i8>, ptr } %load, 0
  %next.dst = call ptr @llvm.riscv.esp.vst.128.ip.m(<16 x i8> %value, ptr %dst,
                                                    i32 16)
  ret ptr %next.dst
}

declare { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip.m(ptr, i32)
declare ptr @llvm.riscv.esp.vst.128.ip.m(<16 x i8>, ptr, i32)
