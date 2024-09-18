// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### -no-canonical-prefixes -target xtensa-esp-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1-ESP-DEFAULT %s
// CC1-ESP-DEFAULT: clang{{.*}} "-cc1" "-triple" "xtensa-esp-unknown-elf"

// RUN: %clang %s -### -no-canonical-prefixes -target xtensa-esp-elf -mcpu=esp32\
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1-ESP32 %s
// CC1-ESP32: clang{{.*}} "-cc1" "-triple" "xtensa-esp-unknown-elf" {{.*}}"-target-cpu" "esp32"

// RUN: %clang %s -### -no-canonical-prefixes -target xtensa-esp-elf -mcpu=esp32s2\
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1-ESP32S2 %s
// CC1-ESP32S2: clang{{.*}} "-cc1" "-triple" "xtensa-esp-unknown-elf" {{.*}}"-target-cpu" "esp32s2"

// RUN: %clang %s -### -no-canonical-prefixes -target xtensa-esp-elf -mcpu=esp32s3\
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1-ESP32S3 %s
// CC1-ESP32S3: clang{{.*}} "-cc1" "-triple" "xtensa-esp-unknown-elf" {{.*}}"-target-cpu" "esp32s3"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target xtensa-esp-elf --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree -fno-rtti 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-ESP32-BAREMETAL %s

// C-XTENSA-ESP32-BAREMETAL: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-ESP32-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0/no-rtti"
// C-XTENSA-ESP32-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib/no-rtti"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target xtensa-esp-elf --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-ESP32-BAREMETAL-RTTI %s

// C-XTENSA-ESP32-BAREMETAL-RTTI: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-ESP32-BAREMETAL-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0"
// C-XTENSA-ESP32-BAREMETAL-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target xtensa-esp-elf --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree -fno-rtti -mfix-esp32-psram-cache-issue 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-ESP32-BAREMETAL-PSRAM %s

// C-XTENSA-ESP32-BAREMETAL-PSRAM: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-ESP32-BAREMETAL-PSRAM: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0/esp32-psram/no-rtti"
// C-XTENSA-ESP32-BAREMETAL-PSRAM: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib/esp32-psram/no-rtti"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target xtensa-esp-elf --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree -mfix-esp32-psram-cache-issue 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-ESP32-BAREMETAL-PSRAM-RTTI %s

// C-XTENSA-ESP32-BAREMETAL-PSRAM-RTTI: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-ESP32-BAREMETAL-PSRAM-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0/esp32-psram"
// C-XTENSA-ESP32-BAREMETAL-PSRAM-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib/esp32-psram"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target xtensa-esp-elf -mcpu=esp32s2 --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree -fno-rtti 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-ESP32S2-BAREMETAL %s

// C-XTENSA-ESP32S2-BAREMETAL: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s2-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32s2-elf-ld"
// C-XTENSA-ESP32S2-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s2-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32s2-elf{{/|\\\\}}8.4.0/no-rtti"
// C-XTENSA-ESP32S2-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s2-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32s2-elf{{/|\\\\}}lib/no-rtti"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target xtensa-esp-elf -mcpu=esp32s2 --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-ESP32S2-BAREMETAL-RTTI %s

// C-XTENSA-ESP32S2-BAREMETAL-RTTI: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s2-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32s2-elf-ld"
// C-XTENSA-ESP32S2-BAREMETAL-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s2-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32s2-elf{{/|\\\\}}8.4.0"
// C-XTENSA-ESP32S2-BAREMETAL-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s2-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32s2-elf{{/|\\\\}}lib"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target xtensa-esp-elf -mcpu=esp32s3 --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree -fno-rtti 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-ESP32S3-BAREMETAL %s

// C-XTENSA-ESP32S3-BAREMETAL: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s3-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32s3-elf-ld"
// C-XTENSA-ESP32S3-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s3-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32s3-elf{{/|\\\\}}8.4.0/no-rtti"
// C-XTENSA-ESP32S3-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s3-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32s3-elf{{/|\\\\}}lib/no-rtti"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target xtensa-esp-elf -mcpu=esp32s3 --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-ESP32S3-BAREMETAL-RTTI %s

// C-XTENSA-ESP32S3-BAREMETAL-RTTI: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s3-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32s3-elf-ld"
// C-XTENSA-ESP32S3-BAREMETAL-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s3-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32s3-elf{{/|\\\\}}8.4.0"
// C-XTENSA-ESP32S3-BAREMETAL-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32s3-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32s3-elf{{/|\\\\}}lib"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target xtensa-esp-elf -mcpu=esp32 --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree \
// RUN:   --sysroot=%S/Inputs/multilib_xtensa_tree/xtensa-esp32-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-ESP32-SYSROOT-BAREMETAL %s

// C-XTENSA-ESP32-SYSROOT-BAREMETAL: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-ESP32-SYSROOT-BAREMETAL: "--sysroot={{.*}}/Inputs/multilib_xtensa_tree/xtensa-esp32-elf"
// C-XTENSA-ESP32-SYSROOT-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0"
// C-XTENSA-ESP32-SYSROOT-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib"

// RUN: %clang++ %s -### -no-canonical-prefixes \
// RUN:   -target xtensa-esp-elf -mcpu=esp32 -stdlib=libstdc++ --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-XTENSA-ESP32-BAREMETAL %s

// CXX-XTENSA-ESP32-BAREMETAL: "-internal-isystem" "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf/include/c++{{/|\\\\}}8.4.0"
// CXX-XTENSA-ESP32-BAREMETAL: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// CXX-XTENSA-ESP32-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0"
// CXX-XTENSA-ESP32-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib"

// RUN: %clang++ %s -### -no-canonical-prefixes \
// RUN:   -target xtensa-esp-elf -mcpu=esp32 -stdlib=libstdc++ --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree \
// RUN:   --sysroot=%S/Inputs/multilib_xtensa_tree/xtensa-esp32-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-XTENSA-ESP32-SYSROOT-BAREMETAL %s

// CXX-XTENSA-ESP32-SYSROOT-BAREMETAL: "-internal-isystem" "{{.*}}Inputs/multilib_xtensa_tree/xtensa-esp32-elf/include/c++/8.4.0"
// CXX-XTENSA-ESP32-SYSROOT-BAREMETAL: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// CXX-XTENSA-ESP32-SYSROOT-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0"
// CXX-XTENSA-ESP32-SYSROOT-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib"
