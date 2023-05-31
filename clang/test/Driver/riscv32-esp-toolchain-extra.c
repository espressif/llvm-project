// A basic clang -cc1 command-line, and simple environment check.

// The tests here are similar to those in xtensa-toolchain.c, however
// these tests need to create symlinks to test directory trees in order to
// set up the environment and therefore shell support is required.
// REQUIRES: shell, xtensa-registered-target
// UNSUPPORTED: system-windows

// Compiler-rt multilibs are located at '$INSTALLDIR/lib/clang/15.0.0/<triple>/mcpu/'.
// At this moment multilib feature for compiler-rt is supported only when GCC installation with the same multilib structure is found.
// It is safe because ESP toolchain still depends on libstdc++ which is part of GCC installation.
// When libc++ wil be supported by toolchain the dependency on GCC multilibs will be removed.

// RUN: rm -rf %t
// RUN: mkdir -p %t/multilib_riscv_esp_elf_sdk/bin
// RUN: ln -s %clang %t/multilib_riscv_esp_elf_sdk/bin/clang
// RUN: ln -s %S/Inputs/multilib_riscv_esp_elf_sdk/bin/riscv32-esp-elf-ld %t/multilib_riscv_esp_elf_sdk/bin/riscv32-esp-elf-ld
// RUN: ln -s %S/Inputs/multilib_riscv_esp_elf_sdk/lib %t/multilib_riscv_esp_elf_sdk/lib
// RUN: ln -s %S/Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf %t/multilib_riscv_esp_elf_sdk/riscv32-esp-elf

// RUN: %t/multilib_riscv_esp_elf_sdk/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_riscv_esp_elf_sdk \
// RUN:    -resource-dir=%t/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0 \
// RUN:    --target=riscv32-esp-elf -march=rv32i -mabi=ilp32 --rtlib=compiler-rt -fuse-ld= -fno-rtti 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32I-RTLIB-COMPILERRT-NORTTI %s

// C-RV32I-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32I-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib/rv32i/ilp32/no-rtti{{/|\\\\}}crt0.o"
// C-RV32I-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32i/ilp32/no-rtti/lib/clang_rt.crtbegin.o"
// C-RV32I-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32i/ilp32/no-rtti/lib/libclang_rt.builtins.a"
// C-RV32I-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32i/ilp32/no-rtti/lib/clang_rt.crtend.o"

// RUN: %t/multilib_riscv_esp_elf_sdk/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_riscv_esp_elf_sdk \
// RUN:    -resource-dir=%t/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0 \
// RUN:    --target=riscv32-esp-elf -march=rv32i -mabi=ilp32 --rtlib=compiler-rt -fuse-ld= 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32I-RTLIB-COMPILERRT-RTTI %s

// C-RV32I-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32I-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib/rv32i/ilp32{{/|\\\\}}crt0.o"
// C-RV32I-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32i/ilp32/lib/clang_rt.crtbegin.o"
// C-RV32I-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32i/ilp32/lib/libclang_rt.builtins.a"
// C-RV32I-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32i/ilp32/lib/clang_rt.crtend.o"

// RUN: %t/multilib_riscv_esp_elf_sdk/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_riscv_esp_elf_sdk \
// RUN:    -resource-dir=%t/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0 \
// RUN:    --target=riscv32-esp-elf -march=rv32imc -mabi=ilp32 --rtlib=compiler-rt -fuse-ld= -fno-rtti 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32IMC-RTLIB-COMPILERRT-NORTTI %s

// C-RV32IMC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32IMC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib/rv32imc/ilp32/no-rtti{{/|\\\\}}crt0.o"
// C-RV32IMC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imc/ilp32/no-rtti/lib/clang_rt.crtbegin.o"
// C-RV32IMC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imc/ilp32/no-rtti/lib/libclang_rt.builtins.a"
// C-RV32IMC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imc/ilp32/no-rtti/lib/clang_rt.crtend.o"

// RUN: %t/multilib_riscv_esp_elf_sdk/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_riscv_esp_elf_sdk \
// RUN:    -resource-dir=%t/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0 \
// RUN:    --target=riscv32-esp-elf -march=rv32imc -mabi=ilp32 --rtlib=compiler-rt -fuse-ld= 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32IMC-RTLIB-COMPILERRT-RTTI %s

// C-RV32IMC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32IMC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib/rv32imc/ilp32{{/|\\\\}}crt0.o"
// C-RV32IMC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imc/ilp32/lib/clang_rt.crtbegin.o"
// C-RV32IMC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imc/ilp32/lib/libclang_rt.builtins.a"
// C-RV32IMC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imc/ilp32/lib/clang_rt.crtend.o"

// RUN: %t/multilib_riscv_esp_elf_sdk/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_riscv_esp_elf_sdk \
// RUN:    -resource-dir=%t/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0 \
// RUN:    --target=riscv32-esp-elf -march=rv32imac -mabi=ilp32 --rtlib=compiler-rt -fuse-ld= -fno-rtti 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32IMAC-RTLIB-COMPILERRT-NORTTI %s

// C-RV32IMAC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32IMAC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib/rv32imac/ilp32/no-rtti{{/|\\\\}}crt0.o"
// C-RV32IMAC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imac/ilp32/no-rtti/lib/clang_rt.crtbegin.o"
// C-RV32IMAC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imac/ilp32/no-rtti/lib/libclang_rt.builtins.a"
// C-RV32IMAC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imac/ilp32/no-rtti/lib/clang_rt.crtend.o"

// RUN: %t/multilib_riscv_esp_elf_sdk/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_riscv_esp_elf_sdk \
// RUN:    -resource-dir=%t/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0 \
// RUN:    --target=riscv32-esp-elf -march=rv32imac -mabi=ilp32 --rtlib=compiler-rt -fuse-ld= 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32IMAC-RTLIB-COMPILERRT-RTTI %s

// C-RV32IMAC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32IMAC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib/rv32imac/ilp32{{/|\\\\}}crt0.o"
// C-RV32IMAC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imac/ilp32/lib/clang_rt.crtbegin.o"
// C-RV32IMAC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imac/ilp32/lib/libclang_rt.builtins.a"
// C-RV32IMAC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imac/ilp32/lib/clang_rt.crtend.o"

// RUN: %t/multilib_riscv_esp_elf_sdk/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_riscv_esp_elf_sdk \
// RUN:    -resource-dir=%t/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0 \
// RUN:    --target=riscv32-esp-elf -march=rv32imafc -mabi=ilp32f --rtlib=compiler-rt -fuse-ld= -fno-rtti 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32IMAFC-RTLIB-COMPILERRT-NORTTI %s

// C-RV32IMAFC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32IMAFC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib/rv32imafc/ilp32f/no-rtti{{/|\\\\}}crt0.o"
// C-RV32IMAFC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imafc/ilp32f/no-rtti/lib/clang_rt.crtbegin.o"
// C-RV32IMAFC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imafc/ilp32f/no-rtti/lib/libclang_rt.builtins.a"
// C-RV32IMAFC-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imafc/ilp32f/no-rtti/lib/clang_rt.crtend.o"

// RUN: %t/multilib_riscv_esp_elf_sdk/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_riscv_esp_elf_sdk \
// RUN:    -resource-dir=%t/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0 \
// RUN:    --target=riscv32-esp-elf -march=rv32imafc -mabi=ilp32f --rtlib=compiler-rt -fuse-ld= 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32IMAFC-RTLIB-COMPILERRT-RTTI %s

// C-RV32IMAFC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32IMAFC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib/rv32imafc/ilp32f{{/|\\\\}}crt0.o"
// C-RV32IMAFC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imafc/ilp32f/lib/clang_rt.crtbegin.o"
// C-RV32IMAFC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imafc/ilp32f/lib/libclang_rt.builtins.a"
// C-RV32IMAFC-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_riscv_esp_elf_sdk/lib/clang/15.0.0/riscv32-esp-elf/generic-rv32/rv32imafc/ilp32f/lib/clang_rt.crtend.o"
