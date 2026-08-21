; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR

; Same espvm vsat intrinsic: +xespv selects ESP_VSAT_*_2P2.

; MIR-DAG: ESP_VSAT_S8_2P2
; MIR-DAG: ESP_VSAT_U8_2P2
; MIR-DAG: ESP_VSAT_S16_2P2
; MIR-DAG: ESP_VSAT_U16_2P2
; MIR-DAG: ESP_VSAT_S32_2P2
; MIR-DAG: ESP_VSAT_U32_2P2

define void @test_vsat_s8_xespv(ptr %src, ptr %dst, i32 %rs1, i32 %rs2) {
; ASM-LABEL: test_vsat_s8_xespv:
; ASM:       esp.vsat.s8
entry:
  %vld = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev = extractvalue { <16 x i8>, ptr } %vld, 0
  %r = call <16 x i8> @llvm.riscv.esp.vsat.s8(<16 x i8> %ev, i32 %rs1, i32 %rs2)
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %r, ptr %dst, i32 16)
  ret void
}

define void @test_vsat_u8_xespv(ptr %src, ptr %dst, i32 %rs1, i32 %rs2) {
; ASM-LABEL: test_vsat_u8_xespv:
; ASM:       esp.vsat.u8
entry:
  %vld = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev = extractvalue { <16 x i8>, ptr } %vld, 0
  %r = call <16 x i8> @llvm.riscv.esp.vsat.u8(<16 x i8> %ev, i32 %rs1, i32 %rs2)
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %r, ptr %dst, i32 16)
  ret void
}

define void @test_vsat_s16_xespv(ptr %src, ptr %dst, i32 %rs1, i32 %rs2) {
; ASM-LABEL: test_vsat_s16_xespv:
; ASM:       esp.vsat.s16
entry:
  %vld = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev = extractvalue { <16 x i8>, ptr } %vld, 0
  %bc = bitcast <16 x i8> %ev to <8 x i16>
  %r = call <8 x i16> @llvm.riscv.esp.vsat.s16(<8 x i16> %bc, i32 %rs1, i32 %rs2)
  %out = bitcast <8 x i16> %r to <16 x i8>
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %out, ptr %dst, i32 16)
  ret void
}

define void @test_vsat_u16_xespv(ptr %src, ptr %dst, i32 %rs1, i32 %rs2) {
; ASM-LABEL: test_vsat_u16_xespv:
; ASM:       esp.vsat.u16
entry:
  %vld = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev = extractvalue { <16 x i8>, ptr } %vld, 0
  %bc = bitcast <16 x i8> %ev to <8 x i16>
  %r = call <8 x i16> @llvm.riscv.esp.vsat.u16(<8 x i16> %bc, i32 %rs1, i32 %rs2)
  %out = bitcast <8 x i16> %r to <16 x i8>
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %out, ptr %dst, i32 16)
  ret void
}

define void @test_vsat_s32_xespv(ptr %src, ptr %dst, i32 %rs1, i32 %rs2) {
; ASM-LABEL: test_vsat_s32_xespv:
; ASM:       esp.vsat.s32
entry:
  %vld = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev = extractvalue { <16 x i8>, ptr } %vld, 0
  %bc = bitcast <16 x i8> %ev to <4 x i32>
  %r = call <4 x i32> @llvm.riscv.esp.vsat.s32(<4 x i32> %bc, i32 %rs1, i32 %rs2)
  %out = bitcast <4 x i32> %r to <16 x i8>
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %out, ptr %dst, i32 16)
  ret void
}

define void @test_vsat_u32_xespv(ptr %src, ptr %dst, i32 %rs1, i32 %rs2) {
; ASM-LABEL: test_vsat_u32_xespv:
; ASM:       esp.vsat.u32
entry:
  %vld = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev = extractvalue { <16 x i8>, ptr } %vld, 0
  %bc = bitcast <16 x i8> %ev to <4 x i32>
  %r = call <4 x i32> @llvm.riscv.esp.vsat.u32(<4 x i32> %bc, i32 %rs1, i32 %rs2)
  %out = bitcast <4 x i32> %r to <16 x i8>
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %out, ptr %dst, i32 16)
  ret void
}

declare { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr, i32)
declare ptr @llvm.riscv.esp.vst.128.ip(<16 x i8>, ptr, i32)
declare <16 x i8> @llvm.riscv.esp.vsat.s8(<16 x i8>, i32, i32)
declare <16 x i8> @llvm.riscv.esp.vsat.u8(<16 x i8>, i32, i32)
declare <8 x i16> @llvm.riscv.esp.vsat.s16(<8 x i16>, i32, i32)
declare <8 x i16> @llvm.riscv.esp.vsat.u16(<8 x i16>, i32, i32)
declare <4 x i32> @llvm.riscv.esp.vsat.s32(<4 x i32>, i32, i32)
declare <4 x i32> @llvm.riscv.esp.vsat.u32(<4 x i32>, i32, i32)
