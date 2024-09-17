; RUN: llc -mtriple=xtensa -mcpu=esp32 -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=XTENSA %s

declare float @llvm.sqrt.f32(float)

define float @sqrt_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI0_0, sqrtf
; XTENSA-LABEL: sqrt_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI0_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n
;
  %1 = call float @llvm.sqrt.f32(float %a)
  ret float %1
}

declare float @llvm.powi.f32(float, i32)

define float @powi_f32(float %a, i32 %b) nounwind {
; XTENSA:    	.literal .LCPI1_0, __powisf2
; XTENSA-LABEL: powi_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI1_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  mov.n	a11, a3
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.powi.f32(float %a, i32 %b)
  ret float %1
}

declare float @llvm.sin.f32(float)

define float @sin_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI2_0, sinf
; XTENSA-LABEL:    	sin_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI2_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.sin.f32(float %a)
  ret float %1
}

declare float @llvm.cos.f32(float)

define float @cos_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI3_0, cosf
; XTENSA-LABEL:    	cos_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI3_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.cos.f32(float %a)
  ret float %1
}

declare float @llvm.pow.f32(float, float)

define float @pow_f32(float %a, float %b) nounwind {
; XTENSA:    	.literal .LCPI4_0, powf
; XTENSA-LABEL:     pow_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI4_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  mov.n	a11, a3
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n
  %1 = call float @llvm.pow.f32(float %a, float %b)
  ret float %1
}

declare float @llvm.exp.f32(float)

define float @exp_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI5_0, expf
; XTENSA-LABEL:    	exp_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI5_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n
  %1 = call float @llvm.exp.f32(float %a)
  ret float %1
}

declare float @llvm.exp2.f32(float)

define float @exp2_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI6_0, exp2
; XTENSA-LABEL:    	exp2_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI6_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n
  %1 = call float @llvm.exp2.f32(float %a)
  ret float %1
}

declare float @llvm.log.f32(float)

define float @log_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI7_0, log
; XTENSA-LABEL:    	log_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI7_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n
  %1 = call float @llvm.log.f32(float %a)
  ret float %1
}

declare float @llvm.log10.f32(float)

define float @log10_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI8_0, log10
; XTENSA-LABEL:    	log10_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI8_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n
  %1 = call float @llvm.log10.f32(float %a)
  ret float %1
}

declare float @llvm.log2.f32(float)

define float @log2_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI9_0, log2
; XTENSA-LABEL:    	log2_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI9_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n
  %1 = call float @llvm.log2.f32(float %a)
  ret float %1
}

declare float @llvm.fma.f32(float, float, float)

define float @fma_f32(float %a, float %b, float %c) nounwind {
; XTENSA-LABEL:    	fma_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  wfr	f8, a3
; XTENSA-NEXT:	  wfr	f9, a2
; XTENSA-NEXT:	  wfr	f10, a4
; XTENSA-NEXT:	  madd.s	f10, f9, f8
; XTENSA-NEXT:	  rfr	a2, f10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %1
}

declare float @llvm.minnum.f32(float, float)

define float @minnum_f32(float %a, float %b) nounwind {
; XTENSA:    	.literal .LCPI11_0, fminf
; XTENSA-LABEL:     minnum_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI11_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  mov.n	a11, a3
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.minnum.f32(float %a, float %b)
  ret float %1
}

declare float @llvm.maxnum.f32(float, float)

define float @maxnum_f32(float %a, float %b) nounwind {
; XTENSA:    	.literal .LCPI12_0, fmaxf
; XTENSA-LABEL:     maxnum_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI12_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  mov.n	a11, a3
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.maxnum.f32(float %a, float %b)
  ret float %1
}

declare float @llvm.fabs.f32(float)

define float @fabs_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI13_0, 2147483647
; XTENSA-LABEL:    	fabs_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI13_0
; XTENSA-NEXT:	  and	a2, a2, a8
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.fabs.f32(float %a)
  ret float %1
}

declare float @llvm.copysign.f32(float, float)

define float @copysign_f32(float %a, float %b) nounwind {
; XTENSA:       .literal .LCPI14_0, -2147483648
; XTENSA:       .literal .LCPI14_1, 2147483647
; XTENSA-LABEL:     copysign_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI14_0
; XTENSA-NEXT:	  and	a8, a3, a8
; XTENSA-NEXT:	  l32r	a9, .LCPI14_1
; XTENSA-NEXT:	  and	a9, a2, a9
; XTENSA-NEXT:	  wfr	f8, a9
; XTENSA-NEXT:	  movi.n	a9, 0
; XTENSA-NEXT:	  beq	a8, a9, .LBB14_2
; XTENSA-NEXT:	  # %bb.1:
; XTENSA-NEXT:	  neg.s	f8, f8
; XTENSA-NEXT:	  .LBB14_2:
; XTENSA-NEXT:	  rfr	a2, f8
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.copysign.f32(float %a, float %b)
  ret float %1
}

declare float @llvm.floor.f32(float)

define float @floor_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI15_0, floor
; XTENSA-LABEL:    	floor_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI15_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.floor.f32(float %a)
  ret float %1
}

declare float @llvm.ceil.f32(float)

define float @ceil_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI16_0, ceil
; XTENSA-LABEL:    	ceil_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI16_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.ceil.f32(float %a)
  ret float %1
}

declare float @llvm.trunc.f32(float)

define float @trunc_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI17_0, trunc
; XTENSA-LABEL:    	trunc_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI17_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.trunc.f32(float %a)
  ret float %1
}

declare float @llvm.rint.f32(float)

define float @rint_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI18_0, rint
; XTENSA-LABEL:    	rint_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI18_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.rint.f32(float %a)
  ret float %1
}

declare float @llvm.nearbyint.f32(float)

define float @nearbyint_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI19_0, nearbyint
; XTENSA-LABEL:    	nearbyint_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI19_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.nearbyint.f32(float %a)
  ret float %1
}

declare float @llvm.round.f32(float)

define float @round_f32(float %a) nounwind {
; XTENSA:    	.literal .LCPI20_0, round
; XTENSA-LABEL:    	round_f32:
; XTENSA:       # %bb.0:
; XTENSA-NEXT:	  entry	a1, 32
; XTENSA-NEXT:	  l32r	a8, .LCPI20_0
; XTENSA-NEXT:	  mov.n	a10, a2
; XTENSA-NEXT:	  callx8	a8
; XTENSA-NEXT:	  mov.n	a2, a10
; XTENSA-NEXT:	  retw.n

  %1 = call float @llvm.round.f32(float %a)
  ret float %1
}
