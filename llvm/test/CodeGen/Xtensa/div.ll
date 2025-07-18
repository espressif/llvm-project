; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc -mtriple=xtensa --mcpu=generic -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=XTENSA %s

define i32 @udiv(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: udiv:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    l32r a8, .LCPI0_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i32 %a, %b
  ret i32 %1
}

define i32 @udiv_constant(i32 %a) nounwind {
; XTENSA-LABEL: udiv_constant:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    movi a3, 5
; XTENSA-NEXT:    l32r a8, .LCPI1_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i32 %a, 5
  ret i32 %1
}

define i32 @udiv_pow2(i32 %a) nounwind {
; XTENSA-LABEL: udiv_pow2:
; XTENSA:         srli a2, a2, 3
; XTENSA-NEXT:    ret
  %1 = udiv i32 %a, 8
  ret i32 %1
}

define i32 @udiv_constant_lhs(i32 %a) nounwind {
; XTENSA-LABEL: udiv_constant_lhs:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    or a3, a2, a2
; XTENSA-NEXT:    movi a2, 10
; XTENSA-NEXT:    l32r a8, .LCPI3_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i32 10, %a
  ret i32 %1
}

define i64 @udiv64(i64 %a, i64 %b) nounwind {
; XTENSA-LABEL: udiv64:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    l32r a8, .LCPI4_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i64 %a, %b
  ret i64 %1
}

define i64 @udiv64_constant(i64 %a) nounwind {
; XTENSA-LABEL: udiv64_constant:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    movi a4, 5
; XTENSA-NEXT:    movi a5, 0
; XTENSA-NEXT:    l32r a8, .LCPI5_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i64 %a, 5
  ret i64 %1
}

define i64 @udiv64_constant_lhs(i64 %a) nounwind {
; XTENSA-LABEL: udiv64_constant_lhs:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    or a5, a3, a3
; XTENSA-NEXT:    or a4, a2, a2
; XTENSA-NEXT:    movi a2, 10
; XTENSA-NEXT:    movi a3, 0
; XTENSA-NEXT:    l32r a8, .LCPI6_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i64 10, %a
  ret i64 %1
}

define i8 @udiv8(i8 %a, i8 %b) nounwind {
; XTENSA-LABEL: udiv8:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    movi a8, 255
; XTENSA-NEXT:    and a2, a2, a8
; XTENSA-NEXT:    and a3, a3, a8
; XTENSA-NEXT:    l32r a8, .LCPI7_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i8 %a, %b
  ret i8 %1
}

define i8 @udiv8_constant(i8 %a) nounwind {
; XTENSA-LABEL: udiv8_constant:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    movi a8, 255
; XTENSA-NEXT:    and a2, a2, a8
; XTENSA-NEXT:    movi a3, 5
; XTENSA-NEXT:    l32r a8, .LCPI8_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i8 %a, 5
  ret i8 %1
}

define i8 @udiv8_pow2(i8 %a) nounwind {
; XTENSA-LABEL: udiv8_pow2:
; XTENSA:         movi a8, 248
; XTENSA-NEXT:    and a8, a2, a8
; XTENSA-NEXT:    srli a2, a8, 3
; XTENSA-NEXT:    ret
  %1 = udiv i8 %a, 8
  ret i8 %1
}

define i8 @udiv8_constant_lhs(i8 %a) nounwind {
; XTENSA-LABEL: udiv8_constant_lhs:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    movi a8, 255
; XTENSA-NEXT:    and a3, a2, a8
; XTENSA-NEXT:    movi a2, 10
; XTENSA-NEXT:    l32r a8, .LCPI10_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i8 10, %a
  ret i8 %1
}

define i16 @udiv16(i16 %a, i16 %b) nounwind {
; XTENSA-LABEL: udiv16:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    l32r a8, .LCPI11_0
; XTENSA-NEXT:    and a2, a2, a8
; XTENSA-NEXT:    and a3, a3, a8
; XTENSA-NEXT:    l32r a8, .LCPI11_1
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i16 %a, %b
  ret i16 %1
}

define i16 @udiv16_constant(i16 %a) nounwind {
; XTENSA-LABEL: udiv16_constant:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    l32r a8, .LCPI12_0
; XTENSA-NEXT:    and a2, a2, a8
; XTENSA-NEXT:    movi a3, 5
; XTENSA-NEXT:    l32r a8, .LCPI12_1
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = udiv i16 %a, 5
  ret i16 %1
}

define i16 @udiv16_pow2(i16 %a) nounwind {
; XTENSA-LABEL: udiv16_pow2:
; XTENSA:         l32r a8, .LCPI13_0
; XTENSA-NEXT:    and a8, a2, a8
; XTENSA-NEXT:    srli a2, a8, 3
; XTENSA-NEXT:    ret
  %1 = udiv i16 %a, 8
  ret i16 %1
}

define i32 @sdiv(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: sdiv:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    l32r a8, .LCPI14_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i32 %a, %b
  ret i32 %1
}

define i32 @sdiv_constant_lhs(i32 %a) nounwind {
; XTENSA-LABEL: sdiv_constant_lhs:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    or a3, a2, a2
; XTENSA-NEXT:    movi a2, -10
; XTENSA-NEXT:    l32r a8, .LCPI15_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i32 -10, %a
  ret i32 %1
}

define i64 @sdiv64(i64 %a, i64 %b) nounwind {
; XTENSA-LABEL: sdiv64:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    l32r a8, .LCPI16_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i64 %a, %b
  ret i64 %1
}

define i64 @sdiv64_constant(i64 %a) nounwind {
; XTENSA-LABEL: sdiv64_constant:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    movi a4, 5
; XTENSA-NEXT:    movi a5, 0
; XTENSA-NEXT:    l32r a8, .LCPI17_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i64 %a, 5
  ret i64 %1
}

define i64 @sdiv64_constant_lhs(i64 %a) nounwind {
; XTENSA-LABEL: sdiv64_constant_lhs:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    or a5, a3, a3
; XTENSA-NEXT:    or a4, a2, a2
; XTENSA-NEXT:    movi a2, 10
; XTENSA-NEXT:    movi a3, 0
; XTENSA-NEXT:    l32r a8, .LCPI18_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i64 10, %a
  ret i64 %1
}


define i64 @sdiv64_sext_operands(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: sdiv64_sext_operands:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    or a4, a3, a3
; XTENSA-NEXT:    srai a3, a2, 31
; XTENSA-NEXT:    srai a5, a4, 31
; XTENSA-NEXT:    l32r a8, .LCPI19_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sext i32 %a to i64
  %2 = sext i32 %b to i64
  %3 = sdiv i64 %1, %2
  ret i64 %3
}

define i8 @sdiv8(i8 %a, i8 %b) nounwind {
; XTENSA-LABEL: sdiv8:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    slli a8, a2, 24
; XTENSA-NEXT:    srai a2, a8, 24
; XTENSA-NEXT:    slli a8, a3, 24
; XTENSA-NEXT:    srai a3, a8, 24
; XTENSA-NEXT:    l32r a8, .LCPI20_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i8 %a, %b
  ret i8 %1
}

define i8 @sdiv8_constant(i8 %a) nounwind {
; XTENSA-LABEL: sdiv8_constant:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    slli a8, a2, 24
; XTENSA-NEXT:    srai a2, a8, 24
; XTENSA-NEXT:    movi a3, 5
; XTENSA-NEXT:    l32r a8, .LCPI21_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i8 %a, 5
  ret i8 %1
}

define i8 @sdiv8_pow2(i8 %a) nounwind {
; XTENSA-LABEL: sdiv8_pow2:
; XTENSA:         slli a8, a2, 24
; XTENSA-NEXT:    srai a8, a8, 31
; XTENSA-NEXT:    movi a9, 7
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a2, a8
; XTENSA-NEXT:    slli a8, a8, 24
; XTENSA-NEXT:    srai a2, a8, 27
; XTENSA-NEXT:    ret
  %1 = sdiv i8 %a, 8
  ret i8 %1
}

define i8 @sdiv8_constant_lhs(i8 %a) nounwind {
; XTENSA-LABEL: sdiv8_constant_lhs:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    slli a8, a2, 24
; XTENSA-NEXT:    srai a3, a8, 24
; XTENSA-NEXT:    movi a2, -10
; XTENSA-NEXT:    l32r a8, .LCPI23_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i8 -10, %a
  ret i8 %1
}

define i16 @sdiv16(i16 %a, i16 %b) nounwind {
; XTENSA-LABEL: sdiv16:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    slli a8, a2, 16
; XTENSA-NEXT:    srai a2, a8, 16
; XTENSA-NEXT:    slli a8, a3, 16
; XTENSA-NEXT:    srai a3, a8, 16
; XTENSA-NEXT:    l32r a8, .LCPI24_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i16 %a, %b
  ret i16 %1
}

define i16 @sdiv16_constant(i16 %a) nounwind {
; XTENSA-LABEL: sdiv16_constant:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    slli a8, a2, 16
; XTENSA-NEXT:    srai a2, a8, 16
; XTENSA-NEXT:    movi a3, 5
; XTENSA-NEXT:    l32r a8, .LCPI25_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i16 %a, 5
  ret i16 %1
}

define i16 @sdiv16_constant_lhs(i16 %a) nounwind {
; XTENSA-LABEL: sdiv16_constant_lhs:
; XTENSA:         addi a8, a1, -16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    s32i a0, a1, 0 # 4-byte Folded Spill
; XTENSA-NEXT:    slli a8, a2, 16
; XTENSA-NEXT:    srai a3, a8, 16
; XTENSA-NEXT:    movi a2, -10
; XTENSA-NEXT:    l32r a8, .LCPI26_0
; XTENSA-NEXT:    callx0 a8
; XTENSA-NEXT:    l32i a0, a1, 0 # 4-byte Folded Reload
; XTENSA-NEXT:    addi a8, a1, 16
; XTENSA-NEXT:    or a1, a8, a8
; XTENSA-NEXT:    ret
  %1 = sdiv i16 -10, %a
  ret i16 %1
}

define i32 @sdiv_pow2(i32 %a) nounwind {
; XTENSA-LABEL: sdiv_pow2:
; XTENSA:         srai a8, a2, 31
; XTENSA-NEXT:    extui a8, a8, 29, 3
; XTENSA-NEXT:    add a8, a2, a8
; XTENSA-NEXT:    srai a2, a8, 3
; XTENSA-NEXT:    ret
  %1 = sdiv i32 %a, 8
  ret i32 %1
}

define i32 @sdiv_pow2_2(i32 %a) nounwind {
; XTENSA-LABEL: sdiv_pow2_2:
; XTENSA:         srai a8, a2, 31
; XTENSA-NEXT:    extui a8, a8, 16, 16
; XTENSA-NEXT:    add a8, a2, a8
; XTENSA-NEXT:    srai a2, a8, 16
; XTENSA-NEXT:    ret
  %1 = sdiv i32 %a, 65536
  ret i32 %1
}

define i16 @sdiv16_pow2(i16 %a) nounwind {
; XTENSA-LABEL: sdiv16_pow2:
; XTENSA:         slli a8, a2, 16
; XTENSA-NEXT:    srai a8, a8, 31
; XTENSA-NEXT:    movi a9, 7
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a2, a8
; XTENSA-NEXT:    slli a8, a8, 16
; XTENSA-NEXT:    srai a2, a8, 19
; XTENSA-NEXT:    ret
  %1 = sdiv i16 %a, 8
  ret i16 %1
}
