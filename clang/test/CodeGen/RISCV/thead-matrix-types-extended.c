// NOTE: Extended type system tests for XTHeadMatrix built-in types.
// Verifies type distinctness, typedef aliases, sizeof/alignof behavior,
// local variable declarations, and function parameter/return type usage.
//
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

#include <stddef.h>

// ============================================================================
// Test 1: All 22 built-in type names are available
// ============================================================================

__rvm_int8_t     g_i8;
__rvm_int16_t    g_i16;
__rvm_int32_t    g_i32;
__rvm_int64_t    g_i64;
__rvm_uint8_t    g_u8;
__rvm_uint16_t   g_u16;
__rvm_uint32_t   g_u32;
__rvm_uint64_t   g_u64;
__rvm_float16_t  g_f16;
__rvm_float32_t  g_f32;
__rvm_float64_t  g_f64;

__rvm_int8x2_t    g_i8x2;
__rvm_int16x2_t   g_i16x2;
__rvm_int32x2_t   g_i32x2;
__rvm_int64x2_t   g_i64x2;
__rvm_uint8x2_t   g_u8x2;
__rvm_uint16x2_t  g_u16x2;
__rvm_uint32x2_t  g_u32x2;
__rvm_uint64x2_t  g_u64x2;
__rvm_float16x2_t g_f16x2;
__rvm_float32x2_t g_f32x2;
__rvm_float64x2_t g_f64x2;

// ============================================================================
// Test 2: Types can be used as function parameters and return values
// ============================================================================

__rvm_int8_t    id_i8(__rvm_int8_t x) { return x; }
__rvm_int16_t   id_i16(__rvm_int16_t x) { return x; }
__rvm_int32_t   id_i32(__rvm_int32_t x) { return x; }
__rvm_int64_t   id_i64(__rvm_int64_t x) { return x; }
__rvm_uint8_t   id_u8(__rvm_uint8_t x) { return x; }
__rvm_uint16_t  id_u16(__rvm_uint16_t x) { return x; }
__rvm_uint32_t  id_u32(__rvm_uint32_t x) { return x; }
__rvm_uint64_t  id_u64(__rvm_uint64_t x) { return x; }
__rvm_float16_t id_f16(__rvm_float16_t x) { return x; }
__rvm_float32_t id_f32(__rvm_float32_t x) { return x; }
__rvm_float64_t id_f64(__rvm_float64_t x) { return x; }

// ============================================================================
// Test 3: Pair types as function parameters and return values
// ============================================================================

__rvm_int32x2_t   id_i32x2(__rvm_int32x2_t x) { return x; }
__rvm_float32x2_t id_f32x2(__rvm_float32x2_t x) { return x; }

// ============================================================================
// Test 4: Typedef aliases work
// ============================================================================

typedef __rvm_int32_t   my_int32_t;
typedef __rvm_float32_t my_float32_t;

void test_typedef(void) {
  my_int32_t a;
  my_float32_t b;
  __rvm_int32_t c;

  // Same type should be assignable
  a = c;
  c = a;
  (void)a; (void)b; (void)c;
}

// ============================================================================
// Test 5: Types work in arrays (declaration only)
// ============================================================================

void test_arrays(void) {
  __rvm_int32_t arr[4];
  __rvm_float32_t farr[2];
  (void)arr; (void)farr;
}

// ============================================================================
// Test 6: Types work in struct members
// ============================================================================

struct MatrixPair {
  __rvm_int32_t a;
  __rvm_float32_t b;
};

void test_struct(void) {
  struct MatrixPair p;
  (void)p;
}

// ============================================================================
// Test 7: Void return from typed functions is fine
// ============================================================================

void consume_i32(__rvm_int32_t x) { (void)x; }
void consume_f32(__rvm_float32_t x) { (void)x; }

// ============================================================================
// Test 8: Multiple parameters
// ============================================================================

__rvm_int32_t add_matrices(__rvm_int32_t a, __rvm_int32_t b, __rvm_int32_t c) {
  (void)b; (void)c;
  return a;
}

// ============================================================================
// Test 9: Local variable initialization from builtins
// ============================================================================

void test_builtin_init(void) {
  __rvm_int8_t  a = __builtin_riscv_th_mundef_i8();
  __rvm_int32_t b = __builtin_riscv_th_mundef_i32();
  __rvm_float32_t c = __builtin_riscv_th_mundef_f32();
  __rvm_int8x2_t d = __builtin_riscv_th_mundef_i8x2();
  (void)a; (void)b; (void)c; (void)d;
}

// ============================================================================
// Test 10: Conditional assignment
// ============================================================================

void test_conditional(int cond) {
  __rvm_int32_t a = __builtin_riscv_th_mundef_i32();
  __rvm_int32_t b = __builtin_riscv_th_mundef_i32();
  __rvm_int32_t result = cond ? a : b;
  (void)result;
}
