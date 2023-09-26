// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### -no-canonical-prefixes -target riscv32-esp-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_esp_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "riscv32-esp-unknown-elf"

// Test interaction with -fuse-ld=lld, if lld is available.
// RUN: %clang %s -### -no-canonical-prefixes -target riscv32-esp-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_esp_elf_sdk -fuse-ld=lld 2>&1 \
// RUN:   | FileCheck -check-prefix=LLD %s
// LLD: {{(error: invalid linker name in argument '-fuse-ld=lld')|(ld.lld)}}

// rv32imac is the default

// RUN: %clang %s -### -no-canonical-prefixes -target riscv32-esp-elf \
// RUN:   -ffreestanding --rtlib=libgcc --ld-path=riscv32-esp-elf-ld \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_esp_elf_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32IMAC-BAREMETAL-MULTI-ILP32 %s

// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-as"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "-mabi" "ilp32" "-march" "rv32imac"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "--sysroot={{.*}}/Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "-m" "elf32lriscv"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imac/ilp32{{/|\\\\}}crtbegin.o"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf/lib"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group" "-lgcc"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imac/ilp32{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -target riscv32-esp-elf \
// RUN:   -ffreestanding --rtlib=libgcc --ld-path=riscv32-esp-elf-ld \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_esp_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32 %s

// C-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-as"
// C-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "-mabi" "ilp32" "-march" "rv32imac"
// C-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "-m" "elf32lriscv"
// C-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imac/ilp32{{/|\\\\}}crtbegin.o"
// C-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0"
// C-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib"
// C-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group" "-lgcc"
// C-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imac/ilp32{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes -target riscv32-esp-elf \
// RUN:   -ffreestanding -stdlib=libstdc++ --rtlib=libgcc --ld-path=riscv32-esp-elf-ld \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_esp_elf_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV32IMAC-BAREMETAL-MULTI-ILP32 %s

// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "-internal-isystem" "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf/include/c++{{/|\\\\}}8.4.0"
// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-as"
// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "-mabi" "ilp32" "-march" "rv32imac"
// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "--sysroot={{.*}}/Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf"
// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "-m" "elf32lriscv"
// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imac/ilp32{{/|\\\\}}crtbegin.o"
// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0"
// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf/lib"
// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "-lstdc++" "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group" "-lgcc"
// CXX-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imac/ilp32{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes -target riscv32-esp-elf \
// RUN:   -ffreestanding -stdlib=libstdc++ --rtlib=libgcc --ld-path=riscv32-esp-elf-ld \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_esp_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32 %s

// CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "-internal-isystem" "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/include/c++{{/|\\\\}}8.4.0"
// CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-as"
// CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "-mabi" "ilp32" "-march" "rv32imac"
// CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "-m" "elf32lriscv"
// CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imac/ilp32{{/|\\\\}}crtbegin.o"
// CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0"
// CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib"
// CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "-lstdc++" "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group" "-lgcc"
// CXX-RV32IMAC-BAREMETAL-MULTI-NOSYSROOT-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imac/ilp32{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes -target riscv32-esp-elf \
// RUN:   -ffreestanding -stdlib=libstdc++ --rtlib=libgcc --ld-path=riscv32-esp-elf-ld \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_esp_elf_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf \
// RUN:   -fno-rtti 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32 %s

// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "-internal-isystem" "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf/include/c++{{/|\\\\}}8.4.0"
// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-as"
// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "-mabi" "ilp32" "-march" "rv32imac"
// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "--sysroot={{.*}}/Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf"
// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "-m" "elf32lriscv"
// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imac/ilp32/no-rtti{{/|\\\\}}crtbegin.o"
// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0"
// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/riscv32-esp-elf/lib"
// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "-lstdc++" "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group" "-lgcc"
// CXX-RV32IMAC-BAREMETAL-MULTI-NORTTI-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imac/ilp32/no-rtti{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -target riscv32-esp-elf \
// RUN:   -march=rv32i -mabi=ilp32 \
// RUN:   -ffreestanding --rtlib=libgcc --ld-path=riscv32-esp-elf-ld --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_esp_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32I-BAREMETAL-MULTI-ILP32 %s

// C-RV32I-BAREMETAL-MULTI-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-as"
// C-RV32I-BAREMETAL-MULTI-ILP32: "-mabi" "ilp32" "-march" "rv32i"
// C-RV32I-BAREMETAL-MULTI-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32I-BAREMETAL-MULTI-ILP32: "-m" "elf32lriscv"
// C-RV32I-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32i/ilp32{{/|\\\\}}crtbegin.o"
// C-RV32I-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0"
// C-RV32I-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib"
// C-RV32I-BAREMETAL-MULTI-ILP32: "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group" "-lgcc"
// C-RV32I-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32i/ilp32{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -target riscv32-esp-elf \
// RUN:   -march=rv32imc -mabi=ilp32 \
// RUN:   -ffreestanding --rtlib=libgcc --ld-path=riscv32-esp-elf-ld --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_esp_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32IMC-BAREMETAL-MULTI-ILP32 %s

// C-RV32IMC-BAREMETAL-MULTI-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-as"
// C-RV32IMC-BAREMETAL-MULTI-ILP32: "-mabi" "ilp32" "-march" "rv32imc"
// C-RV32IMC-BAREMETAL-MULTI-ILP32: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32IMC-BAREMETAL-MULTI-ILP32: "-m" "elf32lriscv"
// C-RV32IMC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imc/ilp32{{/|\\\\}}crtbegin.o"
// C-RV32IMC-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0"
// C-RV32IMC-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib"
// C-RV32IMC-BAREMETAL-MULTI-ILP32: "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group" "-lgcc"
// C-RV32IMC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imc/ilp32{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -target riscv32-esp-elf \
// RUN:   -march=rv32imafc -mabi=ilp32f \
// RUN:   -ffreestanding --rtlib=libgcc --ld-path=riscv32-esp-elf-ld --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_esp_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32IMAFC-BAREMETAL-MULTI-ILP32F %s

// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-as"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "-mabi" "ilp32f" "-march" "rv32imafc"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "{{.*}}Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-esp-elf-ld"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "-m" "elf32lriscv"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imafc/ilp32f{{/|\\\\}}crtbegin.o"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "-L{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-esp-elf/lib"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group" "-lgcc"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "{{.*}}/Inputs/multilib_riscv_esp_elf_sdk/lib/gcc/riscv32-esp-elf/8.4.0/rv32imafc/ilp32f{{/|\\\\}}crtend.o"

// RUN: %clang -target riscv32-esp-elf %s -emit-llvm -S -o - | FileCheck %s

typedef __builtin_va_list va_list;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __WCHAR_TYPE__ wchar_t;
typedef __WINT_TYPE__ wint_t;


// Check Alignments

// CHECK: @align_c = dso_local global i32 1
int align_c = __alignof(char);

// CHECK: @align_s = dso_local global i32 2
int align_s = __alignof(short);

// CHECK: @align_i = dso_local global i32 4
int align_i = __alignof(int);

// CHECK: @align_wc = dso_local global i32 4
int align_wc = __alignof(wchar_t);

// CHECK: @align_wi = dso_local global i32 4
int align_wi = __alignof(wint_t);

// CHECK: @align_l = dso_local global i32 4
int align_l = __alignof(long);

// CHECK: @align_ll = dso_local global i32 8
int align_ll = __alignof(long long);

// CHECK: @align_p = dso_local global i32 4
int align_p = __alignof(void*);

// CHECK: @align_f = dso_local global i32 4
int align_f = __alignof(float);

// CHECK: @align_d = dso_local global i32 8
int align_d = __alignof(double);

// CHECK: @align_ld = dso_local global i32 16
int align_ld = __alignof(long double);

// CHECK: @align_vl = dso_local global i32 4
int align_vl = __alignof(va_list);

// CHECK: @align_a_c = dso_local global i32 1
int align_a_c = __alignof(_Atomic(char));

// CHECK: @align_a_s = dso_local global i32 2
int align_a_s = __alignof(_Atomic(short));

// CHECK: @align_a_i = dso_local global i32 4
int align_a_i = __alignof(_Atomic(int));

// CHECK: @align_a_wc = dso_local global i32 4
int align_a_wc = __alignof(_Atomic(wchar_t));

// CHECK: @align_a_wi = dso_local global i32 4
int align_a_wi = __alignof(_Atomic(wint_t));

// CHECK: @align_a_l = dso_local global i32 4
int align_a_l = __alignof(_Atomic(long));

// CHECK: @align_a_ll = dso_local global i32 8
int align_a_ll = __alignof(_Atomic(long long));

// CHECK: @align_a_p = dso_local global i32 4
int align_a_p = __alignof(_Atomic(void*));

// CHECK: @align_a_f = dso_local global i32 4
int align_a_f = __alignof(_Atomic(float));

// CHECK: @align_a_d = dso_local global i32 8
int align_a_d = __alignof(_Atomic(double));

// CHECK: @align_a_ld = dso_local global i32 16
int align_a_ld = __alignof(_Atomic(long double));

// CHECK: @align_a_s4 = dso_local global i32 4
int align_a_s4 = __alignof(_Atomic(struct { char s[4]; }));

// CHECK: @align_a_s8 = dso_local global i32 8
int align_a_s8 = __alignof(_Atomic(struct { char s[8]; }));

// CHECK: @align_a_s16 = dso_local global i32 16
int align_a_s16 = __alignof(_Atomic(struct { char s[16]; }));

// CHECK: @align_a_s32 = dso_local global i32 1
int align_a_s32 = __alignof(_Atomic(struct { char s[32]; }));


// Check Sizes

// CHECK: @size_a_c = dso_local global i32 1
int size_a_c = sizeof(_Atomic(char));

// CHECK: @size_a_s = dso_local global i32 2
int size_a_s = sizeof(_Atomic(short));

// CHECK: @size_a_i = dso_local global i32 4
int size_a_i = sizeof(_Atomic(int));

// CHECK: @size_a_wc = dso_local global i32 4
int size_a_wc = sizeof(_Atomic(wchar_t));

// CHECK: @size_a_wi = dso_local global i32 4
int size_a_wi = sizeof(_Atomic(wint_t));

// CHECK: @size_a_l = dso_local global i32 4
int size_a_l = sizeof(_Atomic(long));

// CHECK: @size_a_ll = dso_local global i32 8
int size_a_ll = sizeof(_Atomic(long long));

// CHECK: @size_a_p = dso_local global i32 4
int size_a_p = sizeof(_Atomic(void*));

// CHECK: @size_a_f = dso_local global i32 4
int size_a_f = sizeof(_Atomic(float));

// CHECK: @size_a_d = dso_local global i32 8
int size_a_d = sizeof(_Atomic(double));

// CHECK: @size_a_ld = dso_local global i32 16
int size_a_ld = sizeof(_Atomic(long double));


// Check types

// CHECK: zeroext i8 @check_char()
char check_char() { return 0; }

// CHECK: define dso_local signext i16 @check_short()
short check_short() { return 0; }

// CHECK: define dso_local i32 @check_int()
int check_int() { return 0; }

// CHECK: define dso_local i32 @check_wchar_t()
int check_wchar_t() { return 0; }

// CHECK: define dso_local i32 @check_long()
long check_long() { return 0; }

// CHECK: define dso_local i64 @check_longlong()
long long check_longlong() { return 0; }

// CHECK: define dso_local zeroext i8 @check_uchar()
unsigned char check_uchar() { return 0; }

// CHECK: define dso_local zeroext i16 @check_ushort()
unsigned short check_ushort() { return 0; }

// CHECK: define dso_local i32 @check_uint()
unsigned int check_uint() { return 0; }

// CHECK: define dso_local i32 @check_ulong()
unsigned long check_ulong() { return 0; }

// CHECK: define dso_local i64 @check_ulonglong()
unsigned long long check_ulonglong() { return 0; }

// CHECK: define dso_local i32 @check_size_t()
size_t check_size_t() { return 0; }

// CHECK: define dso_local float @check_float()
float check_float() { return 0; }

// CHECK: define dso_local double @check_double()
double check_double() { return 0; }

// CHECK: define dso_local fp128 @check_longdouble()
long double check_longdouble() { return 0; }
