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
// RUN: mkdir -p %t/multilib_xtensa_tree/bin
// RUN: ln -s %clang %t/multilib_xtensa_tree/bin/clang
// RUN: ln -s %S/Inputs/multilib_xtensa_tree/bin/xtensa-esp32-elf-ld %t/multilib_xtensa_tree/bin/xtensa-esp32-elf-ld
// RUN: ln -s %S/Inputs/multilib_xtensa_tree/bin/xtensa-esp32s2-elf-ld %t/multilib_xtensa_tree/bin/xtensa-esp32s2-elf-ld
// RUN: ln -s %S/Inputs/multilib_xtensa_tree/bin/xtensa-esp32s3-elf-ld %t/multilib_xtensa_tree/bin/xtensa-esp32s3-elf-ld
// RUN: ln -s %S/Inputs/multilib_xtensa_tree/lib %t/multilib_xtensa_tree/lib
// RUN: ln -s %S/Inputs/multilib_xtensa_tree/xtensa-esp32-elf %t/multilib_xtensa_tree/xtensa-esp32-elf
// RUN: ln -s %S/Inputs/multilib_xtensa_tree/xtensa-esp32s2-elf %t/multilib_xtensa_tree/xtensa-esp32s2-elf
// RUN: ln -s %S/Inputs/multilib_xtensa_tree/xtensa-esp32s3-elf %t/multilib_xtensa_tree/xtensa-esp32s3-elf

// RUN: %t/multilib_xtensa_tree/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_xtensa_tree \
// RUN:    -resource-dir=%t/multilib_xtensa_tree/lib/clang/15.0.0 \
// RUN:    --target=xtensa-esp-elf -mcpu=esp32 --rtlib=compiler-rt -fuse-ld= -fno-rtti 2>&1 \
// RUN:    | FileCheck -check-prefix=C-XTENSA-ESP32-RTLIB-COMPILERRT-NORTTI %s

// C-XTENSA-ESP32-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/no-rtti/lib/clang_rt.crtbegin.o"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/no-rtti/lib/libclang_rt.builtins.a"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/no-rtti/lib/clang_rt.crtend.o"

// RUN: %t/multilib_xtensa_tree/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_xtensa_tree \
// RUN:    -resource-dir=%t/multilib_xtensa_tree/lib/clang/15.0.0 \
// RUN:    --target=xtensa-esp-elf -mcpu=esp32 --rtlib=compiler-rt -fuse-ld= 2>&1 \
// RUN:    | FileCheck -check-prefix=C-XTENSA-ESP32-RTLIB-COMPILERRT-RTTI %s

// C-XTENSA-ESP32-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/lib/clang_rt.crtbegin.o"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/lib/libclang_rt.builtins.a"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/lib/clang_rt.crtend.o"

// RUN: %t/multilib_xtensa_tree/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_xtensa_tree \
// RUN:    -resource-dir=%t/multilib_xtensa_tree/lib/clang/15.0.0 \
// RUN:    --target=xtensa-esp-elf -mcpu=esp32 --rtlib=compiler-rt -fuse-ld= -mfix-esp32-psram-cache-issue -fno-rtti 2>&1 \
// RUN:    | FileCheck -check-prefix=C-XTENSA-ESP32-RTLIB-COMPILERRT-PSRAM-NORTTI %s

// C-XTENSA-ESP32-RTLIB-COMPILERRT-PSRAM-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-PSRAM-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/esp32-psram/no-rtti/lib/clang_rt.crtbegin.o"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-PSRAM-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/esp32-psram/no-rtti/lib/libclang_rt.builtins.a"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-PSRAM-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/esp32-psram/no-rtti/lib/clang_rt.crtend.o"

// RUN: %t/multilib_xtensa_tree/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_xtensa_tree \
// RUN:    -resource-dir=%t/multilib_xtensa_tree/lib/clang/15.0.0 \
// RUN:    --target=xtensa-esp-elf -mcpu=esp32 --rtlib=compiler-rt -fuse-ld= -mfix-esp32-psram-cache-issue 2>&1 \
// RUN:    | FileCheck -check-prefix=C-XTENSA-ESP32-RTLIB-COMPILERRT-PSRAM-RTTI %s

// C-XTENSA-ESP32-RTLIB-COMPILERRT-PSRAM-RTTI: "{{.*}}/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-PSRAM-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/esp32-psram/lib/clang_rt.crtbegin.o"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-PSRAM-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/esp32-psram/lib/libclang_rt.builtins.a"
// C-XTENSA-ESP32-RTLIB-COMPILERRT-PSRAM-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32/esp32-psram/lib/clang_rt.crtend.o"

// RUN: %t/multilib_xtensa_tree/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_xtensa_tree \
// RUN:    -resource-dir=%t/multilib_xtensa_tree/lib/clang/15.0.0 \
// RUN:    --target=xtensa-esp-elf -mcpu=esp32s2 --rtlib=compiler-rt -fuse-ld= -fno-rtti 2>&1 \
// RUN:    | FileCheck -check-prefix=C-XTENSA-ESP32S2-RTLIB-COMPILERRT-NORTTI %s

// C-XTENSA-ESP32S2-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/gcc/xtensa-esp32s2-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32s2-elf-ld"
// C-XTENSA-ESP32S2-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s2/no-rtti/lib/clang_rt.crtbegin.o"
// C-XTENSA-ESP32S2-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s2/no-rtti/lib/libclang_rt.builtins.a"
// C-XTENSA-ESP32S2-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s2/no-rtti/lib/clang_rt.crtend.o"

// RUN: %t/multilib_xtensa_tree/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_xtensa_tree \
// RUN:    -resource-dir=%t/multilib_xtensa_tree/lib/clang/15.0.0 \
// RUN:    --target=xtensa-esp-elf -mcpu=esp32s2 --rtlib=compiler-rt -fuse-ld= 2>&1 \
// RUN:    | FileCheck -check-prefix=C-XTENSA-ESP32S2-RTLIB-COMPILERRT-RTTI %s

// C-XTENSA-ESP32S2-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/gcc/xtensa-esp32s2-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32s2-elf-ld"
// C-XTENSA-ESP32S2-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s2/lib/clang_rt.crtbegin.o"
// C-XTENSA-ESP32S2-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s2/lib/libclang_rt.builtins.a"
// C-XTENSA-ESP32S2-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s2/lib/clang_rt.crtend.o"

// RUN: %t/multilib_xtensa_tree/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_xtensa_tree \
// RUN:    -resource-dir=%t/multilib_xtensa_tree/lib/clang/15.0.0 \
// RUN:    --target=xtensa-esp-elf -mcpu=esp32s3 --rtlib=compiler-rt -fuse-ld= -fno-rtti 2>&1 \
// RUN:    | FileCheck -check-prefix=C-XTENSA-ESP32S3-RTLIB-COMPILERRT-NORTTI %s

// C-XTENSA-ESP32S3-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/gcc/xtensa-esp32s3-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32s3-elf-ld"
// C-XTENSA-ESP32S3-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s3/no-rtti/lib/clang_rt.crtbegin.o"
// C-XTENSA-ESP32S3-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s3/no-rtti/lib/libclang_rt.builtins.a"
// C-XTENSA-ESP32S3-RTLIB-COMPILERRT-NORTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s3/no-rtti/lib/clang_rt.crtend.o"

// RUN: %t/multilib_xtensa_tree/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/multilib_xtensa_tree \
// RUN:    -resource-dir=%t/multilib_xtensa_tree/lib/clang/15.0.0 \
// RUN:    --target=xtensa-esp-elf -mcpu=esp32s3 --rtlib=compiler-rt -fuse-ld= 2>&1 \
// RUN:    | FileCheck -check-prefix=C-XTENSA-ESP32S3-RTLIB-COMPILERRT-RTTI %s

// C-XTENSA-ESP32S3-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/gcc/xtensa-esp32s3-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32s3-elf-ld"
// C-XTENSA-ESP32S3-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s3/lib/clang_rt.crtbegin.o"
// C-XTENSA-ESP32S3-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s3/lib/libclang_rt.builtins.a"
// C-XTENSA-ESP32S3-RTLIB-COMPILERRT-RTTI: "{{.*}}/multilib_xtensa_tree/lib/clang/15.0.0/xtensa-esp-elf/esp32s3/lib/clang_rt.crtend.o"
