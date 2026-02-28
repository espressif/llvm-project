// NOTE: Test that XTHeadMatrix built-in types are available and compile correctly.
//
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

// Test that built-in matrix type names are available
__rvm_int8_t     test_i8;
__rvm_int16_t    test_i16;
__rvm_int32_t    test_i32;
__rvm_int64_t    test_i64;
__rvm_uint8_t    test_u8;
__rvm_uint16_t   test_u16;
__rvm_uint32_t   test_u32;
__rvm_uint64_t   test_u64;
__rvm_float16_t  test_f16;
__rvm_float32_t  test_f32;
__rvm_float64_t  test_f64;

// Register-pair types
__rvm_int8x2_t    test_i8x2;
__rvm_int16x2_t   test_i16x2;
__rvm_int32x2_t   test_i32x2;
__rvm_int64x2_t   test_i64x2;
__rvm_uint8x2_t   test_u8x2;
__rvm_uint16x2_t  test_u16x2;
__rvm_uint32x2_t  test_u32x2;
__rvm_uint64x2_t  test_u64x2;
__rvm_float16x2_t test_f16x2;
__rvm_float32x2_t test_f32x2;
__rvm_float64x2_t test_f64x2;

// Test function parameters and return types
__rvm_int32_t identity(__rvm_int32_t x) {
  return x;
}

// Test that types are distinct
void test_type_safety(void) {
  __rvm_int32_t a;
  __rvm_float32_t b;
  (void)a;
  (void)b;
}
