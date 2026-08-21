; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR

; Same espvm vmin LD/ST.INCP intrinsic: +xespv selects ESP_VMIN_*_*_INCP_2P2.

; MIR-DAG: ESP_VMIN_S8_LD_INCP_2P2
; MIR-DAG: ESP_VMIN_S8_ST_INCP_2P2
; MIR-DAG: ESP_VMIN_S16_LD_INCP_2P2
; MIR-DAG: ESP_VMIN_S16_ST_INCP_2P2
; MIR-DAG: ESP_VMIN_S32_LD_INCP_2P2
; MIR-DAG: ESP_VMIN_S32_ST_INCP_2P2
; MIR-DAG: ESP_VMIN_U8_LD_INCP_2P2
; MIR-DAG: ESP_VMIN_U8_ST_INCP_2P2
; MIR-DAG: ESP_VMIN_U16_LD_INCP_2P2
; MIR-DAG: ESP_VMIN_U16_ST_INCP_2P2
; MIR-DAG: ESP_VMIN_U32_LD_INCP_2P2
; MIR-DAG: ESP_VMIN_U32_ST_INCP_2P2

define void @test_vmin_s8_ld_incp_xespv(ptr %src1, ptr %src2) {
; ASM-LABEL: test_vmin_s8_ld_incp_xespv:
; ASM:       esp.vmin.s8.ld.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %v1 = call { <16 x i8>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.s8.ld.incp(<16 x i8> %ev1, <16 x i8> %ev2, ptr %src1)
  ret void
}

define void @test_vmin_s8_st_incp_xespv(ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; ASM-LABEL: test_vmin_s8_st_incp_xespv:
; ASM:       esp.vmin.s8.st.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %vld3 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src3, i32 16)
  %ev3 = extractvalue { <16 x i8>, ptr } %vld3, 0
  %v1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vmin.s8.st.incp(<16 x i8> %ev1, <16 x i8> %ev2, <16 x i8> %ev3, ptr %dst, <16 x i8> %ev3)
  ret void
}

define void @test_vmin_s16_ld_incp_xespv(ptr %src1, ptr %src2) {
; ASM-LABEL: test_vmin_s16_ld_incp_xespv:
; ASM:       esp.vmin.s16.ld.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %bc1 = bitcast <16 x i8> %ev1 to <8 x i16>
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %bc2 = bitcast <16 x i8> %ev2 to <8 x i16>
  %v1 = call { <8 x i16>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.s16.ld.incp(<8 x i16> %bc1, <8 x i16> %bc2, ptr %src1)
  ret void
}

define void @test_vmin_s16_st_incp_xespv(ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; ASM-LABEL: test_vmin_s16_st_incp_xespv:
; ASM:       esp.vmin.s16.st.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %bc1 = bitcast <16 x i8> %ev1 to <8 x i16>
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %bc2 = bitcast <16 x i8> %ev2 to <8 x i16>
  %vld3 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src3, i32 16)
  %ev3 = extractvalue { <16 x i8>, ptr } %vld3, 0
  %.cast = bitcast <16 x i8> %ev3 to <8 x i16>
  %v1 = call { <8 x i16>, ptr } @llvm.riscv.esp.vmin.s16.st.incp(<8 x i16> %bc1, <8 x i16> %bc2, <16 x i8> %ev3, ptr %dst, <8 x i16> %.cast)
  ret void
}

define void @test_vmin_s32_ld_incp_xespv(ptr %src1, ptr %src2) {
; ASM-LABEL: test_vmin_s32_ld_incp_xespv:
; ASM:       esp.vmin.s32.ld.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %bc1 = bitcast <16 x i8> %ev1 to <4 x i32>
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %bc2 = bitcast <16 x i8> %ev2 to <4 x i32>
  %v1 = call { <4 x i32>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.s32.ld.incp(<4 x i32> %bc1, <4 x i32> %bc2, ptr %src1)
  ret void
}

define void @test_vmin_s32_st_incp_xespv(ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; ASM-LABEL: test_vmin_s32_st_incp_xespv:
; ASM:       esp.vmin.s32.st.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %bc1 = bitcast <16 x i8> %ev1 to <4 x i32>
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %bc2 = bitcast <16 x i8> %ev2 to <4 x i32>
  %vld3 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src3, i32 16)
  %ev3 = extractvalue { <16 x i8>, ptr } %vld3, 0
  %.cast = bitcast <16 x i8> %ev3 to <4 x i32>
  %v1 = call { <4 x i32>, ptr } @llvm.riscv.esp.vmin.s32.st.incp(<4 x i32> %bc1, <4 x i32> %bc2, <16 x i8> %ev3, ptr %dst, <4 x i32> %.cast)
  ret void
}

define void @test_vmin_u8_ld_incp_xespv(ptr %src1, ptr %src2) {
; ASM-LABEL: test_vmin_u8_ld_incp_xespv:
; ASM:       esp.vmin.u8.ld.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %v1 = call { <16 x i8>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.u8.ld.incp(<16 x i8> %ev1, <16 x i8> %ev2, ptr %src1)
  ret void
}

define void @test_vmin_u8_st_incp_xespv(ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; ASM-LABEL: test_vmin_u8_st_incp_xespv:
; ASM:       esp.vmin.u8.st.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %vld3 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src3, i32 16)
  %ev3 = extractvalue { <16 x i8>, ptr } %vld3, 0
  %v1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vmin.u8.st.incp(<16 x i8> %ev1, <16 x i8> %ev2, <16 x i8> %ev3, ptr %dst, <16 x i8> %ev3)
  ret void
}

define void @test_vmin_u16_ld_incp_xespv(ptr %src1, ptr %src2) {
; ASM-LABEL: test_vmin_u16_ld_incp_xespv:
; ASM:       esp.vmin.u16.ld.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %bc1 = bitcast <16 x i8> %ev1 to <8 x i16>
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %bc2 = bitcast <16 x i8> %ev2 to <8 x i16>
  %v1 = call { <8 x i16>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.u16.ld.incp(<8 x i16> %bc1, <8 x i16> %bc2, ptr %src1)
  ret void
}

define void @test_vmin_u16_st_incp_xespv(ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; ASM-LABEL: test_vmin_u16_st_incp_xespv:
; ASM:       esp.vmin.u16.st.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %bc1 = bitcast <16 x i8> %ev1 to <8 x i16>
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %bc2 = bitcast <16 x i8> %ev2 to <8 x i16>
  %vld3 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src3, i32 16)
  %ev3 = extractvalue { <16 x i8>, ptr } %vld3, 0
  %.cast = bitcast <16 x i8> %ev3 to <8 x i16>
  %v1 = call { <8 x i16>, ptr } @llvm.riscv.esp.vmin.u16.st.incp(<8 x i16> %bc1, <8 x i16> %bc2, <16 x i8> %ev3, ptr %dst, <8 x i16> %.cast)
  ret void
}

define void @test_vmin_u32_ld_incp_xespv(ptr %src1, ptr %src2) {
; ASM-LABEL: test_vmin_u32_ld_incp_xespv:
; ASM:       esp.vmin.u32.ld.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %bc1 = bitcast <16 x i8> %ev1 to <4 x i32>
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %bc2 = bitcast <16 x i8> %ev2 to <4 x i32>
  %v1 = call { <4 x i32>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.u32.ld.incp(<4 x i32> %bc1, <4 x i32> %bc2, ptr %src1)
  ret void
}

define void @test_vmin_u32_st_incp_xespv(ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; ASM-LABEL: test_vmin_u32_st_incp_xespv:
; ASM:       esp.vmin.u32.st.incp
entry:
  %vld1 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %bc1 = bitcast <16 x i8> %ev1 to <4 x i32>
  %vld2 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %ev2 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %bc2 = bitcast <16 x i8> %ev2 to <4 x i32>
  %vld3 = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src3, i32 16)
  %ev3 = extractvalue { <16 x i8>, ptr } %vld3, 0
  %.cast = bitcast <16 x i8> %ev3 to <4 x i32>
  %v1 = call { <4 x i32>, ptr } @llvm.riscv.esp.vmin.u32.st.incp(<4 x i32> %bc1, <4 x i32> %bc2, <16 x i8> %ev3, ptr %dst, <4 x i32> %.cast)
  ret void
}

declare { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr, i32)
declare { <16 x i8>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.s8.ld.incp(<16 x i8>, <16 x i8>, ptr)
declare { <16 x i8>, ptr } @llvm.riscv.esp.vmin.s8.st.incp(<16 x i8>, <16 x i8>, <16 x i8>, ptr, <16 x i8>)
declare { <8 x i16>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.s16.ld.incp(<8 x i16>, <8 x i16>, ptr)
declare { <8 x i16>, ptr } @llvm.riscv.esp.vmin.s16.st.incp(<8 x i16>, <8 x i16>, <16 x i8>, ptr, <8 x i16>)
declare { <4 x i32>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.s32.ld.incp(<4 x i32>, <4 x i32>, ptr)
declare { <4 x i32>, ptr } @llvm.riscv.esp.vmin.s32.st.incp(<4 x i32>, <4 x i32>, <16 x i8>, ptr, <4 x i32>)
declare { <16 x i8>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.u8.ld.incp(<16 x i8>, <16 x i8>, ptr)
declare { <16 x i8>, ptr } @llvm.riscv.esp.vmin.u8.st.incp(<16 x i8>, <16 x i8>, <16 x i8>, ptr, <16 x i8>)
declare { <8 x i16>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.u16.ld.incp(<8 x i16>, <8 x i16>, ptr)
declare { <8 x i16>, ptr } @llvm.riscv.esp.vmin.u16.st.incp(<8 x i16>, <8 x i16>, <16 x i8>, ptr, <8 x i16>)
declare { <4 x i32>, <16 x i8>, ptr } @llvm.riscv.esp.vmin.u32.ld.incp(<4 x i32>, <4 x i32>, ptr)
declare { <4 x i32>, ptr } @llvm.riscv.esp.vmin.u32.st.incp(<4 x i32>, <4 x i32>, <16 x i8>, ptr, <4 x i32>)
