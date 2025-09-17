//===-- ieee754_sqrtf.S - single precision square root --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

  .text

#ifndef __XTENSA_SOFT_FLOAT__
  .align  4
  .global __ieee754_sqrtf
  .type   __ieee754_sqrtf, @function
__ieee754_sqrtf:
#ifdef __XTENSA_WINDOWED_ABI__
  entry      sp, 16
#endif
  wfr        f0, a2
  sqrt0.s    f1, f0
  const.s    f2, 0
  maddn.s    f2, f1, f1
  nexp01.s   f3, f0
  const.s    f7, 3
  addexp.s   f3, f7
  maddn.s    f7, f2, f3
  nexp01.s   f2, f0
  neg.s      f4, f2
  maddn.s    f1, f7, f1
  const.s    f7, 0
  const.s    f5, 0
  const.s    f6, 0
  maddn.s    f7, f4, f1
  maddn.s    f5, f1, f3
  const.s    f3, 3
  maddn.s    f6, f3, f1
  maddn.s    f2, f7, f7
  maddn.s    f3, f5, f1
  neg.s      f1, f6
  maddn.s    f7, f2, f1
  maddn.s    f6, f3, f6
  mksadj.s   f1, f0
  nexp01.s   f0, f0
  maddn.s    f0, f7, f7
  neg.s      f2, f6
  addexpm.s  f7, f1
  addexp.s   f2, f1
  divn.s     f7, f0, f2
  rfr        a2, f7
#ifdef __XTENSA_WINDOWED_ABI__
  retw
#endif
#endif