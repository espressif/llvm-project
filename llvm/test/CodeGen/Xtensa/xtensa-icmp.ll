; RUN: llc -O1 -mtriple=xtensa -mcpu=esp32 %s -o - | FileCheck %s

define i8 @test_bit(i8 %a) {
; CHECK-LABEL: @test_bit
  %b = and i8 %a, 16
  %bool = icmp eq i8 %b, 0
  br i1 %bool, label %true, label %false
; CHECK: movi.n	a8, 16
; CHECK: and	a8, a2, a8
; CHECK: bnez	a8, .LBB0_2

true:
  ret i8 1

false:
  ret i8 0
}
