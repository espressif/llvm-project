; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR

; Same espvm vabs intrinsic: +xespv selects ESP_VABS_*_2P2.

; MIR-DAG: ESP_VABS_8_2P2
; MIR-DAG: ESP_VABS_16_2P2
; MIR-DAG: ESP_VABS_32_2P2

define void @test_vabs_8_xespv(ptr %src, ptr %dst) {
; ASM-LABEL: test_vabs_8_xespv:
; ASM:       esp.vabs.8
entry:
  %vld = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev = extractvalue { <16 x i8>, ptr } %vld, 0
  %r = call <16 x i8> @llvm.riscv.esp.vabs.8(<16 x i8> %ev)
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %r, ptr %dst, i32 16)
  ret void
}

define void @test_vabs_16_xespv(ptr %src, ptr %dst) {
; ASM-LABEL: test_vabs_16_xespv:
; ASM:       esp.vabs.16
entry:
  %vld = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev = extractvalue { <16 x i8>, ptr } %vld, 0
  %bc = bitcast <16 x i8> %ev to <8 x i16>
  %r = call <8 x i16> @llvm.riscv.esp.vabs.16(<8 x i16> %bc)
  %out = bitcast <8 x i16> %r to <16 x i8>
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %out, ptr %dst, i32 16)
  ret void
}

define void @test_vabs_32_xespv(ptr %src, ptr %dst) {
; ASM-LABEL: test_vabs_32_xespv:
; ASM:       esp.vabs.32
entry:
  %vld = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev = extractvalue { <16 x i8>, ptr } %vld, 0
  %bc = bitcast <16 x i8> %ev to <4 x i32>
  %r = call <4 x i32> @llvm.riscv.esp.vabs.32(<4 x i32> %bc)
  %out = bitcast <4 x i32> %r to <16 x i8>
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %out, ptr %dst, i32 16)
  ret void
}

declare { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr, i32)
declare ptr @llvm.riscv.esp.vst.128.ip(<16 x i8>, ptr, i32)
declare <16 x i8> @llvm.riscv.esp.vabs.8(<16 x i8>)
declare <8 x i16> @llvm.riscv.esp.vabs.16(<8 x i16>)
declare <4 x i32> @llvm.riscv.esp.vabs.32(<4 x i32>)
