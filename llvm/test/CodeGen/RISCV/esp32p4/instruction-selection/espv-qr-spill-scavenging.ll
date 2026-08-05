; RUN: llc -O2 -mtriple=riscv32-esp-unknown-elf -mattr=+xespv2p1,+espv-lowering -verify-machineinstrs < %s -o /dev/null

target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-esp-unknown-elf"

define void @qr_spill_needs_gprpie_scavenging(ptr %0, ptr %1, double %B22) {
BB:
  store <8 x double> <double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF>, ptr %0, align 64
  br label %CF

CF:
  %Cmp33 = fcmp uge double 0.000000e+00, %B22
  br i1 %Cmp33, label %CF, label %CF84

CF84:
  %L49 = load <4 x double>, ptr %0, align 32
  store <16 x double> <double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00, double 0xFFFFFFFFFFFFFFFF>, ptr %1, align 128
  store <4 x double> %L49, ptr null, align 32
  ret void
}
