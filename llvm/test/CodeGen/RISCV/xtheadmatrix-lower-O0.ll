; RUN: llc -O0 -mtriple=riscv64 -mattr=+experimental-xtheadmatrix -verify-machineinstrs < %s \
; RUN:   | FileCheck %s
;
; Test that at -O0, the RISCVLowerMatrixType pass inserts spill/reload
; around matrix definitions to keep live ranges short for fast regalloc.
; The pass inserts msme_internal8 (store) and mlme_internal8 (load) calls.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown"

define void @test_O0_spill(ptr %a_ptr, ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_O0_spill:
; CHECK:        th.mlae32
; CHECK:        th.mlce32
; CHECK:        th.mfmacc.s
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %a = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a_ptr, i64 %stride)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c_ptr, i64 %stride)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %a)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %r, ptr %c_ptr, i64 %stride)
  ret void
}

declare void @llvm.riscv.th.msettilem.i64(i64)
declare void @llvm.riscv.th.msettilek.i64(i64)
declare void @llvm.riscv.th.msettilen.i64(i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr, i64)
declare void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
