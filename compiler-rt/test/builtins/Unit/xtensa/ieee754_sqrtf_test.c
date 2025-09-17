// REQUIRES: xtensa-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t
#define SINGLE_PRECISION
#include "fp_lib.h"
#include "int_lib.h"
#include <math.h>
#include <stdio.h>

#if __xtensa__
extern float __ieee754_sqrtf(float a);

uint32_t test_data[32] = {
  0x1f411656,
  0x1f088887,
  0x7fc00000,
  0x7fc00000,
  0x4fb5d274,
  0x7fc00000,
  0x4bb53869,
  0x7fc00000,
  0x4ab511d2,
  0x35350b63,
  0x2a800246,
  0x7fc00000,
  0x426f77ec,
  0x7fc00000,
  0x7fc00000,
  0x541460cd,
  0x39ea0f5a,
  0x2cd13a2c,
  0x7fc00000,
  0x43054444,
  0x3160c8cb,
  0x7fc00000,
  0x4423b1e1,
  0x31f40eb0,
  0x28d6dcdd,
  0x2446a9bc,
  0x22066202,
  0x20e172a9,
  0x204c71ae,
  0x20088887,
  0x1fc11656,
  0x1f888887
};

int test__ieee754_sqrtf(){
    if (__ieee754_sqrtf(-0.00000000) != -0.00000000)
      return 1;
    if (__ieee754_sqrtf(0.00000000) != 0.00000000)
      return 1;
    if (!isnan(__ieee754_sqrtf(NAN)))
      return 1;
    if (__ieee754_sqrtf(INFINITY) != INFINITY)
      return 1;
    if (__ieee754_sqrtf(4.0) != 2.0)
      return 1;

    //Check some simple tests usign precalculated data
    uint32_t x = 0x123456;
    for (int i = 0; i < 32; i++) {
      if (toRep(__ieee754_sqrtf(fromRep(x))) != test_data[i])
        return 1;
      x = (x >> 1) | ((x & 1) << 31) ;
    }
    return 0;
}
#endif


int main()
{
#if __xtensa__
    if (test__ieee754_sqrtf())
        return 1;
#else
    printf("skipped\n");
#endif

    return 0;
}
