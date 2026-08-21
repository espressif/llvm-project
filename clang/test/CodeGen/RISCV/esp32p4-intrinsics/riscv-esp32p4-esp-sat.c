// ESP.SAT — BuiltinRISCVESPV + unified sat_m (coverage lost when broad
// riscv-esp32p4.c was deleted).
// RUN: %clang_cc1 -triple riscv32 -target-feature +xespv2p1 -emit-llvm -O1 -o - %s \
// RUN: | FileCheck %s

// CHECK-LABEL: define{{.*}} @test_esp_sat(
// CHECK: call i32 @llvm.riscv.esp.sat(
int test_esp_sat(unsigned int rs0, unsigned int rs1, unsigned int rsd) {
  return __builtin_riscv_esp_sat(rs0, rs1, rsd);
}

// CHECK-LABEL: define{{.*}} @test_esp_sat_m(
// CHECK: call i32 @llvm.riscv.esp.sat.m(
unsigned int test_esp_sat_m(unsigned int rs0, unsigned int rs1, unsigned int rsd) {
  return __builtin_riscv_esp_sat_m(rs0, rs1, rsd);
}
