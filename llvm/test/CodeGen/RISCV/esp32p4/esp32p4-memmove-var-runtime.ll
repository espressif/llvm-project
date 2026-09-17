; RUN: opt -S -mtriple=riscv32-esp-unknown-elf -passes=riscv-esp32-p4-memmove -riscv-esp32-p4-memmove=true < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-esp-unknown-elf"

; size >= 128: large.backward splits into rem → Blocks16 → Blocks128 (not a single memmove).
define void @test_var_large_runtime(ptr align 16 %a, ptr align 16 %b, i32 %size) {
; CHECK-LABEL: define void @test_var_large_runtime(
; CHECK:       [[IS_MEDIUM:%.*]] = icmp ult i32 [[SIZE:%.*]], 128
; CHECK:       [[BLOCKS128:%.*]] = udiv i32 [[SIZE]], 128
; CHECK:       [[REM128:%.*]] = urem i32 [[SIZE]], 128
; CHECK:       [[BLOCKS16:%.*]] = udiv i32 [[REM128]], 16
; CHECK:       [[REMAINDER:%.*]] = urem i32 [[REM128]], 16
; CHECK:       step1.Remainder
; CHECK:       step2.Blocks16
; CHECK:       step3.Blocks128
; CHECK:       Blocks16.header
; CHECK:       Blocks128.body
; CHECK:       call {{.*}} @llvm.riscv.esp.vld.128.ip(
;
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr align 16 %a, ptr align 16 %b, i32 %size, i1 false)
  ret void
}
declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i1)
