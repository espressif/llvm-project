// UNSUPPORTED: system-windows

//////////////////// RISCV /////////////////////////

// IMAC is default and it re-uses IMC sub-dirs

// RUN: rm -rf %t
// RUN: mkdir -p %t/basic_riscv32_esp_tree/bin
// RUN: ln -s %clang %t/basic_riscv32_esp_tree/bin/clang
// RUN: ln -s %S/Inputs/basic_riscv32_esp_tree/bin/ld.lld %t/basic_riscv32_esp_tree/bin/ld.lld
// RUN: ln -s %S/Inputs/basic_riscv32_esp_tree/bin/riscv32-esp-elf-clang-as %t/basic_riscv32_esp_tree/bin/riscv32-esp-elf-clang-as
// RUN: ln -s %S/Inputs/basic_riscv32_esp_tree/bin/riscv32-esp-elf-clang-ld %t/basic_riscv32_esp_tree/bin/riscv32-esp-elf-clang-ld
// RUN: ln -s %S/Inputs/basic_riscv32_esp_tree/lib %t/basic_riscv32_esp_tree/lib

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC %s
// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -march=rv32imac -mabi=ilp32 \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC %s
// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -rtlib=compiler-rt \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC %s
// CHECK-ESP-RV32IMAC:      "-cc1" "-triple" "riscv32-esp-unknown-elf"
// CHECK-ESP-RV32IMAC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32IMAC-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32IMAC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMAC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMAC-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-RV32IMAC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}include"
// CHECK-ESP-RV32IMAC-SAME: "-x" "c++" "{{.*}}baremetal-esp.cpp"
// CHECK-ESP-RV32IMAC-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32IMAC-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-RV32IMAC-SAME: "-m" "elf32lriscv"
// CHECK-ESP-RV32IMAC-SAME: "-o" "a.out"
// CHECK-ESP-RV32IMAC-SAME: "-X" "{{.*}}.o"
// CHECK-ESP-RV32IMAC-SAME: "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-ESP-RV32IMAC-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}lib"
// CHECK-ESP-RV32IMAC-SAME: "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group"
// CHECK-ESP-RV32IMAC-SAME: "-lclang_rt.builtins"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf -fno-integrated-as \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-FORCEAS %s
// CHECK-ESP-RV32IMAC-FORCEAS:      "-cc1" "-triple" "riscv32-esp-unknown-elf"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}include"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-x" "c++" "{{.*}}baremetal-esp.cpp"
// CHECK-ESP-RV32IMAC-FORCEAS-NEXT: riscv32-esp-elf-clang-as{{(.exe)?}}" "-o" "{{.*}}.o" "-c" "{{.*}}.s" "-march=rv32imac" "-mabi=ilp32"
// CHECK-ESP-RV32IMAC-FORCEAS-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-m" "elf32lriscv"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-o" "a.out"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-X" "{{.*}}.o"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}lib"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group"
// CHECK-ESP-RV32IMAC-FORCEAS-SAME: "-lclang_rt.builtins"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf --ld-path=riscv32-esp-elf-clang-ld \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-FORCELD %s
// CHECK-ESP-RV32IMAC-FORCELD: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32IMAC-FORCELD: riscv32-esp-elf-clang-ld{{(.exe)?}}"
// CHECK-ESP-RV32IMAC-FORCELD-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-RV32IMAC-FORCELD-SAME: "-m" "elf32lriscv"
// CHECK-ESP-RV32IMAC-FORCELD-SAME: "-o" "a.out"
// CHECK-ESP-RV32IMAC-FORCELD-SAME: "-X" "{{.*}}.o"
// CHECK-ESP-RV32IMAC-FORCELD-SAME: "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-ESP-RV32IMAC-FORCELD-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}lib"
// CHECK-ESP-RV32IMAC-FORCELD-SAME: "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group"
// CHECK-ESP-RV32IMAC-FORCELD-SAME: "-lclang_rt.builtins"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -nostdlibinc -nobuiltininc \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-LIBINC %s
// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -nostdinc \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-LIBINC %s
// CHECK-ESP-RV32IMAC-LIBINC-NOT: "-internal-isystem"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -rtlib=libgcc \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-LIBGCC %s
// CHECK-ESP-RV32IMAC-LIBGCC-NOT: "-lclang_rt.builtins"
// CHECK-ESP-RV32IMAC-LIBGCC: "-lgcc"

// RUN: %t/basic_riscv32_esp_tree/bin/clang --driver-mode=g++ %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-DEFAULTSTDCXX %s
// RUN: %t/basic_riscv32_esp_tree/bin/clang --driver-mode=g++ %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -stdlib=libstdc++ \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-DEFAULTSTDCXX %s
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}include"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-m" "elf32lriscv"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-o" "a.out"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-X" "{{.*}}.o"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}lib"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-lstdc++"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group"
// CHECK-ESP-RV32IMAC-DEFAULTSTDCXX-SAME: "-lclang_rt.builtins"

// RUN: %t/basic_riscv32_esp_tree/bin/clang --driver-mode=g++ %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -stdlib=libc++ \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-LIBCXX %s
// CHECK-ESP-RV32IMAC-LIBCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32IMAC-LIBCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}{{[^v].*}}"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}include"
// CHECK-ESP-RV32IMAC-LIBCXX: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-m" "elf32lriscv"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-o" "a.out"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-X" "{{.*}}.o"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}lib"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group"
// CHECK-ESP-RV32IMAC-LIBCXX-SAME: "-lclang_rt.builtins"

// RUN: %t/basic_riscv32_esp_tree/bin/clang --driver-mode=g++ %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -nodefaultlibs \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-NDL %s
// CHECK-ESP-RV32IMAC-NDL: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32IMAC-NDL-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32IMAC-NDL: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32IMAC-NDL-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}lib"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -march=rv32i -mabi=ilp32 \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32I %s
// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -march=rv32ic -mabi=ilp32 \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32I %s
// CHECK-ESP-RV32I:      "-cc1" "-triple" "riscv32-esp-unknown-elf"
// CHECK-ESP-RV32I-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32I-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32I-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32i-zicsr-zifencei_ilp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32I-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32I-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-RV32I-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32i-zicsr-zifencei_ilp32{{[/\\]+}}include"
// CHECK-ESP-RV32I-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32I-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-RV32I-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32i-zicsr-zifencei_ilp32{{[/\\]+}}lib"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf -fno-integrated-as \
// RUN:     -march=rv32i -mabi=ilp32 \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32I-FORCEAS %s
// CHECK-ESP-RV32I-FORCEAS: riscv32-esp-elf-clang-as{{(.exe)?}}" "-o" "{{.*}}.o" "-c" "{{.*}}.s" "-march=rv32i" "-mabi=ilp32"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -march=rv32i -mabi=ilp32 \
// RUN:     -fno-rtti \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32I-NORTTI %s
// CHECK-ESP-RV32I-NORTTI:      "-cc1" "-triple" "riscv32-esp-unknown-elf"
// CHECK-ESP-RV32I-NORTTI-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32I-NORTTI-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32I-NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32i-zicsr-zifencei_ilp32_no-rtti{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32I-NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32I-NORTTI-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-RV32I-NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32i-zicsr-zifencei_ilp32_no-rtti{{[/\\]+}}include"
// CHECK-ESP-RV32I-NORTTI-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32I-NORTTI-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32i-zicsr-zifencei_ilp32_no-rtti{{[/\\]+}}lib"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -march=rv32im -mabi=ilp32 \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMC %s
// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -march=rv32imc -mabi=ilp32 \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMC %s
// CHECK-ESP-RV32IMC:      "-cc1" "-triple" "riscv32-esp-unknown-elf"
// CHECK-ESP-RV32IMC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32IMC-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32IMC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imc-zicsr-zifencei_ilp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMC-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-RV32IMC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imc-zicsr-zifencei_ilp32{{[/\\]+}}include"
// CHECK-ESP-RV32IMC-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32IMC-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imc-zicsr-zifencei_ilp32{{[/\\]+}}lib"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf \
// RUN:     -march=rv32imc -mabi=ilp32 \
// RUN:     -fno-rtti \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMC-NORTTI %s
// CHECK-ESP-RV32IMC-NORTTI:      "-cc1" "-triple" "riscv32-esp-unknown-elf"
// CHECK-ESP-RV32IMC-NORTTI-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32IMC-NORTTI-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32IMC-NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imc-zicsr-zifencei_ilp32_no-rtti{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMC-NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMC-NORTTI-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-RV32IMC-NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imc-zicsr-zifencei_ilp32_no-rtti{{[/\\]+}}include"
// CHECK-ESP-RV32IMC-NORTTI-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32IMC-NORTTI-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imc-zicsr-zifencei_ilp32_no-rtti{{[/\\]+}}lib"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf -fno-integrated-as \
// RUN:     -march=rv32im -mabi=ilp32 \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IM-FORCEAS %s
// CHECK-ESP-RV32IM-FORCEAS: riscv32-esp-elf-clang-as{{(.exe)?}}" "-o" "{{.*}}.o" "-c" "{{.*}}.s" "-march=rv32im" "-mabi=ilp32"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf -fno-integrated-as \
// RUN:     -march=rv32imc -mabi=ilp32 \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMC-FORCEAS %s
// CHECK-ESP-RV32IMC-FORCEAS: riscv32-esp-elf-clang-as{{(.exe)?}}" "-o" "{{.*}}.o" "-c" "{{.*}}.s" "-march=rv32imc" "-mabi=ilp32"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf -fno-integrated-as \
// RUN:     -march=rv32imac -mabi=ilp32 \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-FORCEAS2 %s
// CHECK-ESP-RV32IMAC-FORCEAS2: riscv32-esp-elf-clang-as{{(.exe)?}}" "-o" "{{.*}}.o" "-c" "{{.*}}.s" "-march=rv32imac" "-mabi=ilp32"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf -march=rv32imafc -mabi=ilp32f \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAFC %s
// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf -march=rv32imafdc -mabi=ilp32f \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAFC %s
// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf -march=rv32gc -mabi=ilp32f \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAFC %s
// CHECK-ESP-RV32IMAFC:      "-cc1" "-triple" "riscv32-esp-unknown-elf"
// CHECK-ESP-RV32IMAFC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32IMAFC-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32IMAFC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imafc-zicsr-zifencei_ilp32f{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMAFC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMAFC-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-RV32IMAFC-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imafc-zicsr-zifencei_ilp32f{{[/\\]+}}include"
// CHECK-ESP-RV32IMAFC-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32IMAFC-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imafc-zicsr-zifencei_ilp32f{{[/\\]+}}lib"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf -march=rv32imafc -mabi=ilp32f \
// RUN:     -fno-rtti \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAFC-NORTTI %s
// CHECK-ESP-RV32IMAFC-NORTTI:      "-cc1" "-triple" "riscv32-esp-unknown-elf"
// CHECK-ESP-RV32IMAFC-NORTTI-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-RV32IMAFC-NORTTI-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-RV32IMAFC-NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imafc-zicsr-zifencei_ilp32f_no-rtti{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMAFC-NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-RV32IMAFC-NORTTI-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-RV32IMAFC-NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imafc-zicsr-zifencei_ilp32f_no-rtti{{[/\\]+}}include"
// CHECK-ESP-RV32IMAFC-NORTTI-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-RV32IMAFC-NORTTI-SAME: "-L[[SYSROOT]]{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imafc-zicsr-zifencei_ilp32f_no-rtti{{[/\\]+}}lib"

// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 --target=riscv32-esp-elf -fno-integrated-as \
// RUN:     -march=rv32imafc -mabi=ilp32f \
// RUN:     --sysroot=%t/basic_riscv32_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAFC-FORCEAS %s
// CHECK-ESP-RV32IMAFC-FORCEAS: riscv32-esp-elf-clang-as{{(.exe)?}}" "-o" "{{.*}}.o" "-c" "{{.*}}.s" "-march=rv32imafc" "-mabi=ilp32f"

// Check that compiler-rt library without the arch filename suffix will
// be used if present.
// RUN: rm -rf %T/baremetal_clang_rt_noarch
// RUN: mkdir -p %T/baremetal_clang_rt_noarch/lib
// RUN: touch %T/baremetal_clang_rt_noarch/lib/libclang_rt.builtins.a
// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 \
// RUN:     --target=riscv32-esp-elf \
// RUN:     --sysroot=%T/baremetal_clang_rt_noarch \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32_CLANGRT-NOARCH %s
// CHECK-ESP-RV32_CLANGRT-NOARCH: "-lclang_rt.builtins"
// CHECK-ESP-RV32_CLANGRT-NOARCH-NOT: "-lclang_rt.builtins-riscv32"
// RUN: rm -rf %T/baremetal_clang_rt_arch
// RUN: mkdir -p %T/baremetal_clang_rt_arch/lib
// RUN: touch %T/baremetal_clang_rt_arch/lib/libclang_rt.builtins-riscv32.a
// RUN: %t/basic_riscv32_esp_tree/bin/clang %s -### 2>&1 \
// RUN:     --target=riscv32-esp-elf \
// RUN:     --sysroot=%T/baremetal_clang_rt_arch \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32-CLANGRT-ARCH %s
// CHECK-ESP-RV32-CLANGRT-ARCH: "-lclang_rt.builtins-riscv32"
// CHECK-ESP-RV32-CLANGRT-ARCH-NOT: "-lclang_rt.builtins"


//////////////////// XTENSA /////////////////////////

// RUN: rm -rf %t
// RUN: mkdir -p %t/basic_xtensa_esp_tree/bin
// RUN: ln -s %clang %t/basic_xtensa_esp_tree/bin/clang
// RUN: ln -s %S/Inputs/basic_xtensa_esp_tree/bin/ld.lld %t/basic_xtensa_esp_tree/bin/ld.lld
// RUN: ln -s %S/Inputs/basic_xtensa_esp_tree/bin/xtensa-esp32-elf-clang-as %t/basic_xtensa_esp_tree/bin/xtensa-esp32-elf-clang-as
// RUN: ln -s %S/Inputs/basic_xtensa_esp_tree/bin/xtensa-esp32-elf-clang-ld %t/basic_xtensa_esp_tree/bin/xtensa-esp32-elf-clang-ld
// RUN: ln -s %S/Inputs/basic_xtensa_esp_tree/bin/xtensa-esp32s2-elf-clang-as %t/basic_xtensa_esp_tree/bin/xtensa-esp32s2-elf-clang-as
// RUN: ln -s %S/Inputs/basic_xtensa_esp_tree/bin/xtensa-esp32s2-elf-clang-ld %t/basic_xtensa_esp_tree/bin/xtensa-esp32s2-elf-clang-ld
// RUN: ln -s %S/Inputs/basic_xtensa_esp_tree/bin/xtensa-esp32s3-elf-clang-as %t/basic_xtensa_esp_tree/bin/xtensa-esp32s3-elf-clang-as
// RUN: ln -s %S/Inputs/basic_xtensa_esp_tree/bin/xtensa-esp32s3-elf-clang-ld %t/basic_xtensa_esp_tree/bin/xtensa-esp32s3-elf-clang-ld
// RUN: ln -s %S/Inputs/basic_xtensa_esp_tree/lib %t/basic_xtensa_esp_tree/lib

// ESP32 is default

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32 %s
// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf -mcpu=esp32 \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32 %s
// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -rtlib=compiler-rt \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32 %s
// CHECK-ESP-ESP32:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}include"
// CHECK-ESP-ESP32-SAME: "-x" "c++" "{{.*}}baremetal-esp.cpp"
// CHECK-ESP-ESP32-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32-SAME: "-o" "a.out"
// CHECK-ESP-ESP32-SAME: "-X" "{{.*}}.o"
// CHECK-ESP-ESP32-SAME: "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-ESP-ESP32-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}lib"
// CHECK-ESP-ESP32-SAME: "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group"
// CHECK-ESP-ESP32-SAME: "-lclang_rt.builtins"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf -fno-integrated-as \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-FORCEAS %s
// CHECK-ESP-ESP32-FORCEAS:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}include"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-x" "c++" "{{.*}}baremetal-esp.cpp"
// CHECK-ESP-ESP32-FORCEAS-NEXT: xtensa-esp32-elf-clang-as{{(.exe)?}}" "-o" "{{.*}}.o" "-c" "{{.*}}.s"
// CHECK-ESP-ESP32-FORCEAS-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32-FORCEAS-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-o" "a.out"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-X" "{{.*}}.o"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}lib"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group"
// CHECK-ESP-ESP32-FORCEAS-SAME: "-lclang_rt.builtins"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf --ld-path=xtensa-esp32-elf-clang-ld    \
// RUN:     -L some/directory/user/asked/for \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-FORCELD %s
// CHECK-ESP-ESP32-FORCELD: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32-FORCELD-NEXT: xtensa-esp32-elf-clang-ld{{(.exe)?}}"
// CHECK-ESP-ESP32-FORCELD-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32-FORCELD-SAME: "-o" "a.out"
// CHECK-ESP-ESP32-FORCELD-SAME: "-X" "{{.*}}.o"
// CHECK-ESP-ESP32-FORCELD-SAME: "-Lsome{{[/\\]+}}directory{{[/\\]+}}user{{[/\\]+}}asked{{[/\\]+}}for"
// CHECK-ESP-ESP32-FORCELD-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}lib"
// CHECK-ESP-ESP32-FORCELD-SAME: "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group"
// CHECK-ESP-ESP32-FORCELD-SAME: "-lclang_rt.builtins"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -nostdlibinc -nobuiltininc \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-LIBINC %s
// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -nostdinc \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-LIBINC %s
// CHECK-ESP-ESP32-LIBINC-NOT: "-internal-isystem"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -rtlib=libgcc \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-LIBGCC %s
// CHECK-ESP-ESP32-LIBGCC-NOT: "-lclang_rt.builtins"
// CHECK-ESP-ESP32-LIBGCC: "-lgcc"

// RUN: %t/basic_xtensa_esp_tree/bin/clang --driver-mode=g++ %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-DEFAULTSTDCXX %s
// RUN: %t/basic_xtensa_esp_tree/bin/clang --driver-mode=g++ %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -stdlib=libstdc++ \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-DEFAULTSTDCXX %s
// CHECK-ESP-ESP32-DEFAULTSTDCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}include"
// CHECK-ESP-ESP32-DEFAULTSTDCXX: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-o" "a.out"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-X" "{{.*}}.o"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}lib"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-lstdc++"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group"
// CHECK-ESP-ESP32-DEFAULTSTDCXX-SAME: "-lclang_rt.builtins"

// RUN: %t/basic_xtensa_esp_tree/bin/clang --driver-mode=g++ %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -stdlib=libc++ \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-LIBCXX %s
// CHECK-ESP-ESP32-LIBCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32-LIBCXX-NOT: "-internal-isystem" "{{[^"]+}}{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}{{[^v].*}}"
// CHECK-ESP-ESP32-LIBCXX-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32-LIBCXX-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-ESP-ESP32-LIBCXX-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32-LIBCXX-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}include"
// CHECK-ESP-ESP32-LIBCXX: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32-LIBCXX-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32-LIBCXX-SAME: "-o" "a.out"
// CHECK-ESP-ESP32-LIBCXX-SAME: "-X" "{{.*}}.o"
// CHECK-ESP-ESP32-LIBCXX-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}lib"
// CHECK-ESP-ESP32-LIBCXX-SAME: "-lc++" "-lc++abi" "-lunwind"
// CHECK-ESP-ESP32-LIBCXX-SAME: "-lm" "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group"
// CHECK-ESP-ESP32-LIBCXX-SAME: "-lclang_rt.builtins"

// RUN: %t/basic_xtensa_esp_tree/bin/clang --driver-mode=g++ %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -nodefaultlibs \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-NDL %s
// CHECK-ESP-ESP32-NDL: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32-NDL-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32-NDL: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32-NDL-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}lib"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -mcpu=esp32 -fno-rtti \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32_NORTTI %s
// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -fno-rtti \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32_NORTTI %s
// CHECK-ESP-ESP32_NORTTI:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32_NORTTI-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32_NORTTI-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32_no-rtti{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32_NORTTI-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32_no-rtti{{[/\\]+}}include"
// CHECK-ESP-ESP32_NORTTI-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32_NORTTI-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32_NORTTI-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32_no-rtti{{[/\\]+}}lib"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -mcpu=esp32 -mfix-esp32-psram-cache-issue \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32_PSRAM %s
// CHECK-ESP-ESP32_PSRAM:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32_PSRAM-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32_PSRAM-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32_PSRAM-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32_psram{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32_PSRAM-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32_PSRAM-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32_PSRAM-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32_psram{{[/\\]+}}include"
// CHECK-ESP-ESP32_PSRAM-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32_PSRAM-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32_PSRAM-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32_psram{{[/\\]+}}lib"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -mcpu=esp32 -fno-rtti -mfix-esp32-psram-cache-issue \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32_PSRAM_NORTTI %s
// CHECK-ESP-ESP32_PSRAM_NORTTI:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32_PSRAM_NORTTI-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32_PSRAM_NORTTI-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32_PSRAM_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32_psram_no-rtti{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32_PSRAM_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32_PSRAM_NORTTI-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32_PSRAM_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32_psram_no-rtti{{[/\\]+}}include"
// CHECK-ESP-ESP32_PSRAM_NORTTI-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32_PSRAM_NORTTI-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32_PSRAM_NORTTI-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32_psram_no-rtti{{[/\\]+}}lib"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -mcpu=esp32s2 \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32S2 %s
// CHECK-ESP-ESP32S2:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32S2-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32S2-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32S2-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s2{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S2-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S2-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32S2-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s2{{[/\\]+}}include"
// CHECK-ESP-ESP32S2-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32S2-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32S2-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s2{{[/\\]+}}lib"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf -fno-integrated-as \
// RUN:     -mcpu=esp32s2 \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32S2-FORCEAS %s
// CHECK-ESP-ESP32S2-FORCEAS:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32S2-FORCEAS-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32S2-FORCEAS-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32S2-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s2{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S2-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S2-FORCEAS-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32S2-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s2{{[/\\]+}}include"
// CHECK-ESP-ESP32S2-FORCEAS: xtensa-esp32s2-elf-clang-as{{(.exe)?}}" "-o" "{{.*}}.o" "-c" "{{.*}}.s"
// CHECK-ESP-ESP32S2-FORCEAS-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32S2-FORCEAS-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32S2-FORCEAS-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s2{{[/\\]+}}lib"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -mcpu=esp32s2 -fno-rtti \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32S2_NORTTI %s
// CHECK-ESP-ESP32S2_NORTTI:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32S2_NORTTI-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32S2_NORTTI-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32S2_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s2_no-rtti{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S2_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S2_NORTTI-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32S2_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s2_no-rtti{{[/\\]+}}include"
// CHECK-ESP-ESP32S2_NORTTI-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32S2_NORTTI-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32S2_NORTTI-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s2_no-rtti{{[/\\]+}}lib"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -mcpu=esp32s3 \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32S3 %s
// CHECK-ESP-ESP32S3:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32S3-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32S3-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32S3-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s3{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S3-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S3-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32S3-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s3{{[/\\]+}}include"
// CHECK-ESP-ESP32S3-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32S3-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32S3-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s3{{[/\\]+}}lib"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf -fno-integrated-as \
// RUN:     -mcpu=esp32s3 \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32S3-FORCEAS %s
// CHECK-ESP-ESP32S3-FORCEAS:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32S3-FORCEAS-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32S3-FORCEAS-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32S3-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s3{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S3-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S3-FORCEAS-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32S3-FORCEAS-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s3{{[/\\]+}}include"
// CHECK-ESP-ESP32S3-FORCEAS: xtensa-esp32s3-elf-clang-as{{(.exe)?}}" "-o" "{{.*}}.o" "-c" "{{.*}}.s"
// CHECK-ESP-ESP32S3-FORCEAS-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32S3-FORCEAS-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32S3-FORCEAS-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s3{{[/\\]+}}lib"

// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 --target=xtensa-esp-elf \
// RUN:     -mcpu=esp32s3 -fno-rtti \
// RUN:     --sysroot=%t/basic_xtensa_esp_tree/lib/clang-runtimes \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32S3_NORTTI %s
// CHECK-ESP-ESP32S3_NORTTI:      "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32S3_NORTTI-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ESP-ESP32S3_NORTTI-SAME: "-isysroot" "[[SYSROOT:[^"]*]]"
// CHECK-ESP-ESP32S3_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s3_no-rtti{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S3_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECK-ESP-ESP32S3_NORTTI-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// CHECK-ESP-ESP32S3_NORTTI-SAME: "-internal-isystem" "[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s3_no-rtti{{[/\\]+}}include"
// CHECK-ESP-ESP32S3_NORTTI-NEXT: ld.lld{{(.exe)?}}"
// CHECK-ESP-ESP32S3_NORTTI-SAME: "--sysroot=[[SYSROOT]]"
// CHECK-ESP-ESP32S3_NORTTI-SAME: "-L[[SYSROOT]]{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32s3_no-rtti{{[/\\]+}}lib"

// Check that compiler-rt library without the arch filename suffix will
// be used if present.
// RUN: rm -rf %T/baremetal_clang_rt_noarch
// RUN: mkdir -p %T/baremetal_clang_rt_noarch/lib
// RUN: touch %T/baremetal_clang_rt_noarch/lib/libclang_rt.builtins.a
// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 \
// RUN:     --target=xtensa-esp-elf \
// RUN:     --sysroot=%T/baremetal_clang_rt_noarch \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32_CLANGRT-NOARCH %s
// CHECK-ESP-ESP32_CLANGRT-NOARCH: "-lclang_rt.builtins"
// CHECK-ESP-ESP32_CLANGRT-NOARCH-NOT: "-lclang_rt.builtins-xtensa"

// Check that compiler-rt library with the arch filename suffix will be
// used if present.
// RUN: rm -rf %T/baremetal_clang_rt_arch
// RUN: mkdir -p %T/baremetal_clang_rt_arch/lib
// RUN: touch %T/baremetal_clang_rt_arch/lib/libclang_rt.builtins-xtensa.a
// RUN: %t/basic_xtensa_esp_tree/bin/clang %s -### 2>&1 \
// RUN:     --target=xtensa-esp-elf \
// RUN:     --sysroot=%T/baremetal_clang_rt_arch \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-CLANGRT-ARCH %s
// CHECK-ESP-ESP32-CLANGRT-ARCH: "-lclang_rt.builtins-xtensa"
// CHECK-ESP-ESP32-CLANGRT-ARCH-NOT: "-lclang_rt.builtins"
