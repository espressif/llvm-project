; RUN: llc -O2 -mtriple=riscv32-esp-unknown-elf -mattr=+xespv2p1,+espv-lowering %s -o - | FileCheck %s

; Stress-style IR that previously crashed when QR_64 physreg copies were
; lowered through copyPhysReg. RA may not always emit esp.movi.32.{a,q} on
; esp_22.x, so only assert successful codegen here.
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-esp-unknown-elf"

define void @qr64_copy_after_ra(ptr %p, i32 %idx, i1 %cond) {
; CHECK-LABEL: qr64_copy_after_ra:
; CHECK:       ret
entry:
  store <4 x double> zeroinitializer, ptr %p, align 32
  br label %loop0

loop0:
  %elt0 = extractelement <2 x i16> zeroinitializer, i32 %idx
  br i1 %cond, label %loop0, label %loop1

loop1:
  %elt1 = extractelement <4 x i1> zeroinitializer, i32 %idx
  br i1 %cond, label %loop1, label %loop2

loop2:
  %ins = insertelement <2 x i16> zeroinitializer, i16 %elt0, i32 0
  br i1 %cond, label %loop2, label %tail

tail:
  %elt2 = extractelement <1 x double> zeroinitializer, i32 %idx
  %sel = select i1 false, i1 %elt1, i1 false
  br i1 %cond, label %loop1, label %exit

exit:
  store double %elt2, ptr null, align 8
  ret void
}
