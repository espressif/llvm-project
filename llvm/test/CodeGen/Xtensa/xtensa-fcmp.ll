; RUN: llc -O1 -mtriple=xtensa -mcpu=esp32 %s -o - | FileCheck %s

define void @test_fcmp(i32 %x.coerce) {
; CHECK-LABEL: @test_fcmp
entry:
  %0 = bitcast i32 %x.coerce to float
  %cmp = fcmp oeq float %0, 0x7FF0000000000000
  br i1 %cmp, label %if.then, label %if.else
; CHECK: oeq.s	b0, f9, f8
; CHECK: bf	b0, .LBB0_2

if.then:                                          ; preds = %entry
  unreachable

if.else:                                          ; preds = %entry
  unreachable
}

