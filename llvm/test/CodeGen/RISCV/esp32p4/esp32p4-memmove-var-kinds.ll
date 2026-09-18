; RUN: opt -S -mtriple=riscv32-esp-unknown-elf -passes=riscv-esp32-p4-memmove -riscv-esp32-p4-memmove=true < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-esp-unknown-elf"

; These CHECKs lock distinct SIMD shapes per align kind (not unalign.*).

; dst16/src8: vld.l.64 + vst.128
define void @test_var_dst16_src8(ptr align 16 %a, ptr align 8 %b, i32 %size) {
; CHECK-LABEL: define void @test_var_dst16_src8(
; CHECK:       [[IS_SMALL:%.*]] = icmp ult i32 [[SIZE:%.*]], 16
; CHECK:       [[IS_MEDIUM:%.*]] = icmp ult i32 [[SIZE]], 128
; CHECK:       call {{.*}} @llvm.riscv.esp.vld.l.64.ip(
; CHECK:       call {{.*}} @llvm.riscv.esp.vst.128.ip(
; CHECK-NOT:   unalign.is.small
;
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr align 16 %a, ptr align 8 %b, i32 %size, i1 false)
  ret void
}

; dst8/src16: vld.128 + vst.128
define void @test_var_dst8_src16(ptr align 8 %a, ptr align 16 %b, i32 %size) {
; CHECK-LABEL: define void @test_var_dst8_src16(
; CHECK:       [[IS_SMALL:%.*]] = icmp ult i32 [[SIZE:%.*]], 16
; CHECK:       [[IS_MEDIUM:%.*]] = icmp ult i32 [[SIZE]], 128
; CHECK:       call {{.*}} @llvm.riscv.esp.vld.128.ip(
; CHECK:       call {{.*}} @llvm.riscv.esp.vst.128.ip(
; CHECK-NOT:   unalign.is.small
;
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr align 8 %a, ptr align 16 %b, i32 %size, i1 false)
  ret void
}

; dst8/src8: paired 64-bit vld/vst
define void @test_var_dst8_src8(ptr align 8 %a, ptr align 8 %b, i32 %size) {
; CHECK-LABEL: define void @test_var_dst8_src8(
; CHECK:       [[IS_SMALL:%.*]] = icmp ult i32 [[SIZE:%.*]], 16
; CHECK:       [[IS_MEDIUM:%.*]] = icmp ult i32 [[SIZE]], 128
; CHECK:       call {{.*}} @llvm.riscv.esp.vld.l.64.ip(
; CHECK:       call {{.*}} @llvm.riscv.esp.vld.h.64.ip(
; CHECK:       call {{.*}} @llvm.riscv.esp.vst.l.64.ip(
; CHECK:       call {{.*}} @llvm.riscv.esp.vst.h.64.ip(
; CHECK-NOT:   unalign.is.small
;
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr align 8 %a, ptr align 8 %b, i32 %size, i1 false)
  ret void
}

; Wider variable lengths stay intrinsic because runtime offsets are i32.
define void @test_var_wide_size(ptr align 8 %a, ptr align 8 %b, i64 %size) {
; CHECK-LABEL: define void @test_var_wide_size(
; CHECK:       call void @llvm.memmove.p0.p0.i64
; CHECK:       ret void
;
entry:
  tail call void @llvm.memmove.p0.p0.i64(ptr align 8 %a, ptr align 8 %b, i64 %size, i1 false)
  ret void
}

declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i1)
declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)
