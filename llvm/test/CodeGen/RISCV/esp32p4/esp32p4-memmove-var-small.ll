; RUN: opt -S -mtriple=riscv32-esp-unknown-elf -passes=riscv-esp32-p4-memmove -riscv-esp32-p4-memmove=true < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-esp-unknown-elf"

; Variable size in [16,128): medium remainder uses small.back switch (not llvm.memmove).
; Representative case bodies: widths for rem 1/2/4/8 and byte-wise for rem 15.
define void @test_var_medium_remainder(ptr align 16 %a, ptr align 16 %b, i32 %size) {
; CHECK-LABEL: define void @test_var_medium_remainder(
; CHECK:       [[IS_MEDIUM:%.*]] = icmp ult i32 [[SIZE:%.*]], 128
; CHECK:       [[REMAINDER16:%.*]] = urem i32 [[SIZE]], 16
; CHECK:       switch i32 [[REMAINDER16]], label %[[SMALL_BACK_DEFAULT:.*]] [
; CHECK:         i32 15, label %[[SMALL_BACK_CASE_15:.*]]
; CHECK:       small.back.case.1:
; CHECK:       load i8, ptr
; CHECK:       store i8
; CHECK:       small.back.case.2:
; CHECK:       load i16, ptr
; CHECK:       store i16
; CHECK:       small.back.case.4:
; CHECK:       load i32, ptr
; CHECK:       store i32
; CHECK:       small.back.case.8:
; CHECK:       load i64, ptr
; CHECK:       store i64
; CHECK:       small.back.case.15:
; CHECK:       load i8, ptr
; CHECK:       store i8
; CHECK:       [[SIMD_LOOP:.*]]:
; CHECK:       call {{.*}} @llvm.riscv.esp.vld.128.ip(
; CHECK:       call {{.*}} @llvm.riscv.esp.vst.128.ip(
;
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr align 16 %a, ptr align 16 %b, i32 %size, i1 false)
  ret void
}
declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i1)
