; RUN: llc -mtriple=xtensa  -mcpu=esp32 -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=Xtensa %s


define dso_local i32 @test_memptr(i32 noundef %0) local_unnamed_addr #0 {
; Xtensa-LABEL: test_memptr:
; Xtensa:       # %bb.0:
; Xtensa-NEXT:    entry a1, 32
; Xtensa-NEXT:    #APP
; Xtensa-NEXT:    l32i	a2, a2, 0
; Xtensa-NEXT:    #NO_APP
; Xtensa-NEXT:    retw
  %2 = inttoptr i32 %0 to i32*
  %3 = call i32 asm sideeffect "l32i $0, $1", "=r,*m"(i32* elementtype(i32) %2)
  ret i32 %3
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }

