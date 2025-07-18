; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc -mtriple=xtensa -O0 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=XTENSA

define i8 @loadi8_128(i8 %a) {
; XTENSA-LABEL: loadi8_128:
; XTENSA:         entry a1, 416
; XTENSA-NEXT:    movi a8, 127
; XTENSA-NEXT:    movi a9, 128
; XTENSA-NEXT:    and a8, a1, a8
; XTENSA-NEXT:    sub a8, a9, a8
; XTENSA-NEXT:    add.n a1, a1, a8
; XTENSA-NEXT:    movi a8, 128
; XTENSA-NEXT:    add.n a8, a1, a8
; XTENSA-NEXT:    addi a10, a8, 0
; XTENSA-NEXT:    movi.n a11, 0
; XTENSA-NEXT:    movi.n a12, 64
; XTENSA-NEXT:    l32r a8, .LCPI0_0
; XTENSA-NEXT:    callx8 a8
; XTENSA-NEXT:    l8ui a2, a1, 128
; XTENSA-NEXT:    retw.n
    %aligned = alloca i8, align 128
    call void @llvm.memset.p0.i64(ptr noundef nonnull align 64 dereferenceable(64) %aligned, i8 0, i64 64, i1 false)
    %1 = load i8, ptr %aligned, align 128
    ret i8 %1
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
