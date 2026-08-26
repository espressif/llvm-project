; Shared LD.128.USAR / SRC_Q* select 2.1 vs pure 2.2 opcodes.
; Vectors enter/leave via vld/vst.128.ip (or USAR load + vst), not as ABI args/rets.
; RUN: llc -O2 -mtriple=riscv32-esp-unknown-elf -mcpu=esp32p4 -mattr=+espv-lowering \
; RUN:   -stop-after=finalize-isel -verify-machineinstrs %s -o - \
; RUN:   | FileCheck %s --check-prefix=ESPV22
; RUN: llc -O2 -mtriple=riscv32-esp-unknown-elf -mcpu=esp32p4eco4 -mattr=+espv-lowering \
; RUN:   -stop-after=finalize-isel -verify-machineinstrs %s -o - \
; RUN:   | FileCheck %s --check-prefix=ESPV21

; ESPV22-LABEL: name: usar_ip
; ESPV22: ESP_LD_128_USAR_IP_2P2
; ESPV22-NOT: {{[^_]}}ESP_LD_128_USAR_IP{{[^_]}}
; ESPV22: ESP_VST_128_IP_2P2
; ESPV21-LABEL: name: usar_ip
; ESPV21: ESP_LD_128_USAR_IP
; ESPV21-NOT: ESP_LD_128_USAR_IP_2P2
; ESPV21: ESP_VST_128_IP
; ESPV21-NOT: ESP_VST_128_IP_2P2

define void @usar_ip(ptr %dst, ptr %src) {
entry:
  %ld = call { <16 x i8>, ptr, i32 } @llvm.riscv.esp.ld.128.usar.ip(ptr %src, i32 16)
  %v = extractvalue { <16 x i8>, ptr, i32 } %ld, 0
  %st = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %v, ptr %dst, i32 16)
  ret void
}

; ESPV22-LABEL: name: src_q_ld_ip
; ESPV22: ESP_VLD_128_IP_2P2
; ESPV22: ESP_SRC_Q_LD_IP_2P2
; ESPV22: ESP_VST_128_IP_2P2
; ESPV21-LABEL: name: src_q_ld_ip
; ESPV21: ESP_VLD_128_IP
; ESPV21-NOT: ESP_VLD_128_IP_2P2
; ESPV21: ESP_SRC_Q_LD_IP
; ESPV21-NOT: ESP_SRC_Q_LD_IP_2P2
; ESPV21: ESP_VST_128_IP
; ESPV21-NOT: ESP_VST_128_IP_2P2

define void @src_q_ld_ip(ptr %dst, ptr %src_y, ptr %src_w, ptr %p, i32 %sar) {
entry:
  %ld_y = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src_y, i32 16)
  %qy = extractvalue { <16 x i8>, ptr } %ld_y, 0
  %ld_w = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src_w, i32 16)
  %qw = extractvalue { <16 x i8>, ptr } %ld_w, 0
  %r = call { <16 x i8>, <16 x i8>, ptr } @llvm.riscv.esp.src.q.ld.ip(
      i32 %sar, <16 x i8> %qy, <16 x i8> %qw, ptr %p, i32 16)
  %qx = extractvalue { <16 x i8>, <16 x i8>, ptr } %r, 0
  %st = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %qx, ptr %dst, i32 16)
  ret void
}

; ESPV22-LABEL: name: src_q
; ESPV22: ESP_VLD_128_IP_2P2
; ESPV22: ESP_SRC_Q_2P2
; ESPV22: ESP_VST_128_IP_2P2
; ESPV21-LABEL: name: src_q
; ESPV21: ESP_VLD_128_IP
; ESPV21-NOT: ESP_VLD_128_IP_2P2
; ESPV21: ESP_SRC_Q
; ESPV21-NOT: ESP_SRC_Q_2P2
; ESPV21: ESP_VST_128_IP
; ESPV21-NOT: ESP_VST_128_IP_2P2

define void @src_q(ptr %dst, ptr %src_y, ptr %src_w, i32 %sar) {
entry:
  %ld_y = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src_y, i32 16)
  %qy = extractvalue { <16 x i8>, ptr } %ld_y, 0
  %ld_w = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src_w, i32 16)
  %qw = extractvalue { <16 x i8>, ptr } %ld_w, 0
  %r = call <16 x i8> @llvm.riscv.esp.src.q(i32 %sar, <16 x i8> %qy, <16 x i8> %qw)
  %st = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %r, ptr %dst, i32 16)
  ret void
}

declare { <16 x i8>, ptr, i32 } @llvm.riscv.esp.ld.128.usar.ip(ptr, i32)
declare { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr, i32)
declare ptr @llvm.riscv.esp.vst.128.ip(<16 x i8>, ptr, i32)
declare { <16 x i8>, <16 x i8>, ptr } @llvm.riscv.esp.src.q.ld.ip(i32, <16 x i8>, <16 x i8>, ptr, i32)
declare <16 x i8> @llvm.riscv.esp.src.q(i32, <16 x i8>, <16 x i8>)
