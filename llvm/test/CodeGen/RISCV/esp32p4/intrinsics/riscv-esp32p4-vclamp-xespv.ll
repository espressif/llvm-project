; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR

; Same espvm vclamp intrinsic: +xespv selects ESP_VCLAMP_S16_2P2.

; MIR-DAG: ESP_VCLAMP_S16_2P2

define void @test_vclamp_s16_xespv(ptr %src, ptr %dst) {
; ASM-LABEL: test_vclamp_s16_xespv:
; ASM:       esp.vclamp.s16
entry:
  %vld = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev = extractvalue { <16 x i8>, ptr } %vld, 0
  %bc = bitcast <16 x i8> %ev to <8 x i16>
  %r = call <8 x i16> @llvm.riscv.esp.vclamp.s16(<8 x i16> %bc, i32 5)
  %out = bitcast <8 x i16> %r to <16 x i8>
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %out, ptr %dst, i32 16)
  ret void
}

declare { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr, i32)
declare ptr @llvm.riscv.esp.vst.128.ip(<16 x i8>, ptr, i32)
declare <8 x i16> @llvm.riscv.esp.vclamp.s16(<8 x i16>, i32 immarg)
