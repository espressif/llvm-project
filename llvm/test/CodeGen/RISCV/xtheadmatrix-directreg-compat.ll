; RUN: llc -mtriple=riscv64 -mattr=+experimental-xtheadmatrix -verify-machineinstrs < %s \
; RUN:   | FileCheck %s
;
; Backward compatibility test: verify existing DirectReg intrinsics still work
; correctly after the ManagedRA changes. These use ImmArg register indices.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown"

;; Test: DirectReg load-matmul-store still works
define void @test_directreg_matmul(ptr %a, ptr %b, ptr %c, i64 %stride) {
; CHECK-LABEL: test_directreg_matmul:
; CHECK:        th.mlae32 tr0, (a0), a3
; CHECK:        th.mlbe32 tr1, (a1), a3
; CHECK:        th.mlce32 acc0, (a2), a3
; CHECK:        th.mfmacc.s acc0, tr1, tr0
; CHECK:        th.msce32 acc0, (a2), a3
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  call void @llvm.riscv.th.mlae32(i32 0, ptr %a, i64 %stride)
  call void @llvm.riscv.th.mlbe32(i32 1, ptr %b, i64 %stride)
  call void @llvm.riscv.th.mlce32(i32 4, ptr %c, i64 %stride)
  call void @llvm.riscv.th.mfmacc.s(i32 4, i32 1, i32 0)
  call void @llvm.riscv.th.msce32(i32 4, ptr %c, i64 %stride)
  ret void
}

;; Test: DirectReg zero still works
define void @test_directreg_zero() {
; CHECK-LABEL: test_directreg_zero:
; CHECK:        th.mzero acc0
; CHECK:        ret
  call void @llvm.riscv.th.mzero(i32 4)
  ret void
}

;; Test: DirectReg mmov.mm still works
define void @test_directreg_mmov() {
; CHECK-LABEL: test_directreg_mmov:
; CHECK:        th.mmov.mm tr1, tr0
; CHECK:        ret
  call void @llvm.riscv.th.mmov.mm(i32 1, i32 0)
  ret void
}

; DirectReg declarations (existing)
declare void @llvm.riscv.th.msettilem.i64(i64)
declare void @llvm.riscv.th.msettilek.i64(i64)
declare void @llvm.riscv.th.msettilen.i64(i64)
declare void @llvm.riscv.th.mlae32(i32 immarg, ptr, i64)
declare void @llvm.riscv.th.mlbe32(i32 immarg, ptr, i64)
declare void @llvm.riscv.th.mlce32(i32 immarg, ptr, i64)
declare void @llvm.riscv.th.mfmacc.s(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.th.msce32(i32 immarg, ptr, i64)
declare void @llvm.riscv.th.mzero(i32 immarg)
declare void @llvm.riscv.th.mmov.mm(i32 immarg, i32 immarg)
