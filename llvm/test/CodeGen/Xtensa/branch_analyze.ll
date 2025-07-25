; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 2
; RUN: llc -mtriple=xtensa -verify-machineinstrs < %s \
; RUN: | FileCheck -check-prefix=XTENSA %s

declare void @foo() noreturn

; Check reverseBranchCondition and analyzeBranch functions

define i32 @eq(i32 %a, ptr %bptr) {
; XTENSA-LABEL: eq:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    beq a2, a8, .LBB0_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB0_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI0_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @eq_reverse(i32 %a, ptr %bptr) {
; XTENSA-LABEL: eq_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    bne a2, a8, .LBB1_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB1_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI1_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @ne(i32 %a, ptr %bptr) {
; XTENSA-LABEL: ne:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    bne a2, a8, .LBB2_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB2_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI2_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp ne i32 %a, %b
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @ne_reverse(i32 %a, ptr %bptr) {
; XTENSA-LABEL: ne_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    beq a2, a8, .LBB3_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB3_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI3_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp ne i32 %a, %b
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @ult(i32 %a, ptr %bptr) {
; XTENSA-LABEL: ult:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    bltu a2, a8, .LBB4_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB4_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI4_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp ult i32 %a, %b
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @ult_reverse(i32 %a, ptr %bptr) {
; XTENSA-LABEL: ult_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    bgeu a2, a8, .LBB5_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB5_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI5_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp ult i32 %a, %b
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @uge(i32 %a, ptr %bptr) {
; XTENSA-LABEL: uge:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    bgeu a2, a8, .LBB6_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB6_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI6_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp uge i32 %a, %b
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @uge_reverse(i32 %a, ptr %bptr) {
; XTENSA-LABEL: uge_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    bltu a2, a8, .LBB7_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB7_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI7_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp uge i32 %a, %b
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @slt(i32 %a, ptr %bptr) {
; XTENSA-LABEL: slt:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    blt a2, a8, .LBB8_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB8_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI8_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @slt_reverse(i32 %a, ptr %bptr) {
; XTENSA-LABEL: slt_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    bge a2, a8, .LBB9_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB9_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI9_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @sle(i32 %a, ptr %bptr) {
; XTENSA-LABEL: sle:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    bge a8, a2, .LBB10_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB10_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI10_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp sle i32 %a, %b
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @sle_reverse(i32 %a, ptr %bptr) {
; XTENSA-LABEL: sle_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    blt a8, a2, .LBB11_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB11_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI11_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp sle i32 %a, %b
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @sgt(i32 %a, ptr %bptr) {
; XTENSA-LABEL: sgt:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    blt a8, a2, .LBB12_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB12_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI12_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @sgt_reverse(i32 %a, ptr %bptr) {
; XTENSA-LABEL: sgt_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    bge a8, a2, .LBB13_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB13_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI13_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @sge(i32 %a, ptr %bptr) {
; XTENSA-LABEL: sge:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    bge a2, a8, .LBB14_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB14_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI14_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp sge i32 %a, %b
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @sge_reverse(i32 %a, ptr %bptr) {
; XTENSA-LABEL: sge_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    l32i.n a8, a3, 0
; XTENSA-NEXT:    blt a2, a8, .LBB15_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB15_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI15_0
; XTENSA-NEXT:    callx8 a8
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp sge i32 %a, %b
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

; Check some cases of comparing operand with constant.

define i32 @eq_zero(ptr %aptr) {
; XTENSA-LABEL: eq_zero:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    memw
; XTENSA-NEXT:    l32i.n a8, a2, 0
; XTENSA-NEXT:    beqz a8, .LBB16_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB16_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI16_0
; XTENSA-NEXT:    callx8 a8
entry:
  %a = load volatile i32, ptr %aptr
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @eq_zero_reverse(ptr %aptr) {
; XTENSA-LABEL: eq_zero_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    memw
; XTENSA-NEXT:    l32i.n a8, a2, 0
; XTENSA-NEXT:    bnez a8, .LBB17_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB17_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI17_0
; XTENSA-NEXT:    callx8 a8
entry:
  %a = load volatile i32, ptr %aptr
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @ne_zero(ptr %aptr) {
; XTENSA-LABEL: ne_zero:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    memw
; XTENSA-NEXT:    l32i.n a8, a2, 0
; XTENSA-NEXT:    bnez a8, .LBB18_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB18_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI18_0
; XTENSA-NEXT:    callx8 a8
entry:
  %a = load volatile i32, ptr %aptr
  %cmp = icmp ne i32 %a, 0
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @ne_zero_reverse(ptr %aptr) {
; XTENSA-LABEL: ne_zero_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    memw
; XTENSA-NEXT:    l32i.n a8, a2, 0
; XTENSA-NEXT:    beqz a8, .LBB19_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB19_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI19_0
; XTENSA-NEXT:    callx8 a8
entry:
  %a = load volatile i32, ptr %aptr
  %cmp = icmp ne i32 %a, 0
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @slt_zero(ptr %aptr) {
; XTENSA-LABEL: slt_zero:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    memw
; XTENSA-NEXT:    l32i.n a8, a2, 0
; XTENSA-NEXT:    bgez a8, .LBB20_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB20_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI20_0
; XTENSA-NEXT:    callx8 a8
entry:
  %a = load volatile i32, ptr %aptr
  %cmp = icmp slt i32 %a, 0
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @eq_imm(i32 %a) {
; XTENSA-LABEL: eq_imm:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    beqi a2, 1, .LBB21_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB21_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI21_0
; XTENSA-NEXT:    callx8 a8
entry:
  %cmp = icmp eq i32 %a, 1
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @eq_imm_reverse(i32 %a) {
; XTENSA-LABEL: eq_imm_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    bnei a2, 1, .LBB22_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB22_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI22_0
; XTENSA-NEXT:    callx8 a8
entry:
  %cmp = icmp eq i32 %a, 1
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @ne_imm(i32 %a) {
; XTENSA-LABEL: ne_imm:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    beqi a2, 1, .LBB23_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB23_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI23_0
; XTENSA-NEXT:    callx8 a8
entry:
  %cmp = icmp eq i32 %a, 1
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @ne_imm_reverse(i32 %a) {
; XTENSA-LABEL: ne_imm_reverse:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    bnei a2, 1, .LBB24_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB24_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI24_0
; XTENSA-NEXT:    callx8 a8
entry:
  %cmp = icmp eq i32 %a, 1
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @slt_imm(i32 %a) {
; XTENSA-LABEL: slt_imm:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    bgei a2, -1, .LBB25_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB25_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI25_0
; XTENSA-NEXT:    callx8 a8
entry:
  %cmp = icmp slt i32 %a, -1
  br i1 %cmp, label %return, label %callit

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @sge_imm(i32 %a) {
; XTENSA-LABEL: sge_imm:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    beqz a2, .LBB26_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB26_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI26_0
; XTENSA-NEXT:    callx8 a8
entry:
  %cmp = icmp ult i32 %a, 1
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

define i32 @uge_imm(ptr %aptr) {
; XTENSA-LABEL: uge_imm:
; XTENSA:       # %bb.0: # %entry
; XTENSA-NEXT:    entry a1, 32
; XTENSA-NEXT:    memw
; XTENSA-NEXT:    l32i.n a8, a2, 0
; XTENSA-NEXT:    bgeui a8, 2, .LBB27_2
; XTENSA-NEXT:  # %bb.1: # %return
; XTENSA-NEXT:    movi.n a2, 1
; XTENSA-NEXT:    retw.n
; XTENSA-NEXT:  .LBB27_2: # %callit
; XTENSA-NEXT:    l32r a8, .LCPI27_0
; XTENSA-NEXT:    callx8 a8
entry:
  %a = load volatile i32, ptr %aptr
  %cmp = icmp uge i32 %a, 2
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}
