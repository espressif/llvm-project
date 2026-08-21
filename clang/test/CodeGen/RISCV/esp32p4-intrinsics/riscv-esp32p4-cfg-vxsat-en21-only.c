// RUN: %clang_cc1 -triple riscv32 -target-feature +xespv2p1 -emit-llvm -o - %s | FileCheck %s --check-prefix=ESP21
// RUN: %clang_cc1 -triple riscv32 -target-feature +xespv -emit-llvm -o - %s | FileCheck %s --check-prefix=ESP22

// CFG movx builtins are shared; vxsat_en is a PIE 2.1 CFG field only.

// ESP21-LABEL: define{{.*}} @test_movx_cfg
// ESP21: call i32 @llvm.riscv.esp.movx.r.cfg()
// ESP21: call void @llvm.riscv.esp.movx.w.cfg(
void test_movx_cfg(void) {
  unsigned int cfg = __builtin_riscv_esp_movx_r_cfg();
  __builtin_riscv_esp_movx_w_cfg(cfg | 256u);
}

// ESP22-LABEL: define{{.*}} @test_movx_cfg
// ESP22: call i32 @llvm.riscv.esp.movx.r.cfg()
// ESP22: call void @llvm.riscv.esp.movx.w.cfg(
