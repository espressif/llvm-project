// RUN: split-file %s %t
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/correct.c | FileCheck %t/correct.c
// RUN: not %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/bad_vec.c 2>&1 | FileCheck %t/bad_vec.c

//--- correct.c

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));

ae_int32x2 test_ae_int32x2_from_int(int a) {
// CHECK-LABEL: @test_ae_int32x2_from_int(
// CHECK-NEXT:  entry:
// CHECK:       %[[INS:.*]] = insertelement <2 x i32> undef, i32 %a, i64 0
// CHECK:       %[[SHUF:.*]] = shufflevector <2 x i32> %[[INS]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:       ret <2 x i32> %[[SHUF]]
return __builtin_xtensa_ae_int32x2(a);
}

ae_int32x2 test_ae_int32x2_from_ae_int32(ae_int32 a) {
// CHECK-LABEL: @test_ae_int32x2_from_ae_int32(
// CHECK-NEXT:  entry:
// CHECK:       %[[SHUF:.*]] = shufflevector <1 x i32> %a, <1 x i32> poison, <2 x i32> zeroinitializer
// CHECK:       ret <2 x i32> %[[SHUF]]
return __builtin_xtensa_ae_int32x2(a);
}

ae_int32x2 test_ae_int32x2_from_ae_int32x2(ae_int32x2 a) {
// CHECK-LABEL: @test_ae_int32x2_from_ae_int32x2(
// CHECK-NEXT:  entry:
// CHECK:       ret <2 x i32> %a
return __builtin_xtensa_ae_int32x2(a);
}

ae_int32x2 test_ae_int32x2_from_short(short a) {
// CHECK-LABEL: @test_ae_int32x2_from_short(
// CHECK-NEXT:  entry:
// CHECK:       %[[SEXT:.*]] = sext i16 %a to i32
// CHECK:       %[[INS:.*]] = insertelement <2 x i32> undef, i32 %[[SEXT]], i64 0
// CHECK:       %[[SHUF:.*]] = shufflevector <2 x i32> %[[INS]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:       ret <2 x i32> %[[SHUF]]
return __builtin_xtensa_ae_int32x2(a);
}

//--- bad_vec.c

typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));

ae_int32x2 test_ae_int32x2_from_bad_vec(ae_int16x4 a) {
// CHECK: error: passing 'ae_int16x4' {{.*}} to parameter of incompatible type
return __builtin_xtensa_ae_int32x2(a);
}
