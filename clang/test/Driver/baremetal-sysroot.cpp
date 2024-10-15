// REQUIRES: shell
// UNSUPPORTED: system-windows

// Test that when a --sysroot is not provided, driver picks the default
// location correctly if available.

// RUN: rm -rf %T/baremetal_default_sysroot
// RUN: mkdir -p %T/baremetal_default_sysroot/bin
// RUN: mkdir -p %T/baremetal_default_sysroot/lib/clang-runtimes/armv6m-none-eabi
// RUN: ln -s %clang %T/baremetal_default_sysroot/bin/clang

// RUN: %T/baremetal_default_sysroot/bin/clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi --sysroot= \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-C %s
// CHECK-V6M-C: "{{.*}}clang{{.*}}" "-cc1" "-triple" "thumbv6m-none-unknown-eabi"
// CHECK-V6M-C-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}armv6m-none-eabi{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECk-V6M-C-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}armv6m-none-eabi{{[/\\]+}}include"
// CHECK-V6M-C-SAME: "-x" "c++" "{{.*}}baremetal-sysroot.cpp"
// CHECK-V6M-C-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-C-SAME: "-L{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}armv6m-none-eabi{{[/\\]+}}lib"
// CHECK-V6M-C-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-C-SAME: "-o" "{{.*}}.o"

// RUN: rm -rf %T/baremetal_default_sysroot
// RUN: mkdir -p %T/baremetal_default_sysroot/bin
// RUN: mkdir -p %T/baremetal_default_sysroot/lib/clang-runtimes/riscv32-esp-unknown-elf/include/c++/11.2.0
// RUN: mkdir -p %T/baremetal_default_sysroot/lib/clang-runtimes/riscv32-esp-unknown-elf/rv32imac-zicsr-zifencei_ilp32/include/c++/11.2.0
// RUN: echo "MultilibVersion: '1.0'" > %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "Variants:" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "- Dir: riscv32-esp-unknown-elf/rv32imac-zicsr-zifencei_ilp32" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "  Flags:" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "  - --target=riscv32-esp-unknown-elf" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "  - -march=rv32imac_zicsr_zifencei" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "  - -mabi=ilp32" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "Mappings:" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "- Match: -march=rv32imac" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "  Flags:" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "  - -march=rv32imac_zicsr_zifencei" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: ln -s %clang %T/baremetal_default_sysroot/bin/clang

// RUN: %T/baremetal_default_sysroot/bin/clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target riscv32-esp-elf --sysroot= \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-RV32IMAC-C %s
// CHECK-ESP-RV32IMAC-C: "{{.*}}clang{{.*}}" "-cc1" "-triple" "riscv32-esp-unknown-elf"
// CHECK-ESP-RV32IMAC-C-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECk-ESP-RV32IMAC-C-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}include"
// CHECk-ESP-RV32IMAC-C-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}include"
// CHECK-ESP-RV32IMAC-C-SAME: "-x" "c++" "{{.*}}baremetal-sysroot.cpp"
// CHECK-ESP-RV32IMAC-C-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "-m" "elf32lriscv"
// CHECK-ESP-RV32IMAC-C-SAME: "-o" "{{.*}}.o"
// CHECK-ESP-RV32IMAC-C-SAME: "-L{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}riscv32-esp-unknown-elf{{[/\\]+}}rv32imac-zicsr-zifencei_ilp32{{[/\\]+}}lib"
// CHECK-ESP-RV32IMAC-C-SAME: "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group" "-lclang_rt.builtins-riscv32"

// RUN: rm -rf %T/baremetal_default_sysroot
// RUN: mkdir -p %T/baremetal_default_sysroot/bin
// RUN: mkdir -p %T/baremetal_default_sysroot/lib/clang-runtimes/xtensa-esp-unknown-elf/include/c++/11.2.0
// RUN: mkdir -p %T/baremetal_default_sysroot/lib/clang-runtimes/xtensa-esp-unknown-elf/esp32/include/c++/11.2.0
// RUN: echo "MultilibVersion: '1.0'" > %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "Variants:" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "- Dir: xtensa-esp-unknown-elf/esp32" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "  Flags:" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "  - --target=xtensa-esp-unknown-elf" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: echo "  - -mcpu=esp32" >> %T/baremetal_default_sysroot/lib/clang-runtimes/multilib.yaml
// RUN: ln -s %clang %T/baremetal_default_sysroot/bin/clang

// RUN: %T/baremetal_default_sysroot/bin/clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target xtensa-esp-unknown-elf --sysroot= \
// RUN:   | FileCheck --check-prefix=CHECK-ESP-ESP32-C %s
// CHECK-ESP-ESP32-C: "{{.*}}clang{{.*}}" "-cc1" "-triple" "xtensa-esp-unknown-elf"
// CHECK-ESP-ESP32-C-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}11.2.0"
// CHECk-ESP-ESP32-C-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}include"
// CHECk-ESP-ESP32-C-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}include"
// CHECK-ESP-ESP32-C-SAME: "-x" "c++" "{{.*}}baremetal-sysroot.cpp"
// CHECK-ESP-ESP32-C-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}"
// CHECK-ESP-ESP32-C-SAME: "-o" "{{.*}}.o"
// CHECK-ESP-ESP32-C-SAME: "-L{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}xtensa-esp-unknown-elf{{[/\\]+}}esp32{{[/\\]+}}lib"
// CHECK-ESP-ESP32-C-SAME: "--start-group" "-lc" "-lgloss" "-lnosys" "--end-group" "-lclang_rt.builtins-xtensa"
