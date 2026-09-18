; RUN: opt -S -mtriple=riscv32-esp-unknown-elf -passes=riscv-esp32-p4-memmove -riscv-esp32-p4-memmove=true < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-esp-unknown-elf"

; Fully unaligned var: size split + overlap ULE.
define void @test_var_unalign(ptr align 1 %a, ptr align 1 %b, i32 %size) {
; CHECK-LABEL: define void @test_var_unalign(
; CHECK:       [[UNALIGN_IS_SMALL:%.*]] = icmp ult i32 [[SIZE:%.*]], 16
; CHECK:       [[UNALIGN_DST_LEQ_SRC:%.*]] = icmp ule i32 {{.*}}, {{.*}}
; CHECK:       unalign.forward
; CHECK:       unalign.backward
;
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr align 1 %a, ptr align 1 %b, i32 %size, i1 false)
  ret void
}

; Mixed: aligned dst / unaligned src still takes unalign.* (not SIMD kinds).
define void @test_var_dst16_src_unalign(ptr align 16 %a, ptr align 1 %b, i32 %size) {
; CHECK-LABEL: define void @test_var_dst16_src_unalign(
; CHECK:       unalign.is.small
; CHECK:       unalign.forward
; CHECK:       unalign.backward
; CHECK-NOT:   call {{.*}} @llvm.riscv.esp.vld
;
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr align 16 %a, ptr align 1 %b, i32 %size, i1 false)
  ret void
}

; Mixed: 8-byte aligned dst / unaligned src still uses the conservative path.
define void @test_var_dst8_src_unalign(ptr align 8 %a, ptr align 1 %b, i32 %size) {
; CHECK-LABEL: define void @test_var_dst8_src_unalign(
; CHECK:       unalign.is.small
; CHECK:       unalign.forward
; CHECK:       unalign.backward
; CHECK-NOT:   call {{.*}} @llvm.riscv.esp.vld
;
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr align 8 %a, ptr align 1 %b, i32 %size, i1 false)
  ret void
}

; Mixed: unaligned dst / aligned src — forward widens with aligned loads.
define void @test_var_dst_unalign_src16(ptr align 1 %a, ptr align 16 %b, i32 %size) {
; CHECK-LABEL: define void @test_var_dst_unalign_src16(
; CHECK:       unalign.is.small
; CHECK:       unalign.forward
; CHECK:       unalign.backward
; CHECK:       unalign.fwd.w8.head
; CHECK:       [[W8_SRC:%.*]] = getelementptr inbounds i8, ptr %b, i32 [[W8_IDX:%.*]]
; CHECK-NEXT:  [[W8_DST:%.*]] = getelementptr inbounds i8, ptr %a, i32 [[W8_IDX]]
; CHECK-NEXT:  [[W8_LOAD:%.*]] = load i64, ptr [[W8_SRC]], align 8
; CHECK-NEXT:  store i64 [[W8_LOAD]], ptr [[W8_DST]], align 1
; CHECK:       [[W1_SRC:%.*]] = getelementptr inbounds i8, ptr %b, i32 [[W1_IDX:%.*]]
; CHECK-NEXT:  [[W1_DST:%.*]] = getelementptr inbounds i8, ptr %a, i32 [[W1_IDX]]
; CHECK-NEXT:  [[W1_LOAD:%.*]] = load i8, ptr [[W1_SRC]], align 1
; CHECK-NEXT:  store i8 [[W1_LOAD]], ptr [[W1_DST]], align 1
;
entry:
  tail call void @llvm.memmove.p0.p0.i32(ptr align 1 %a, ptr align 16 %b, i32 %size, i1 false)
  ret void
}

declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i1)
