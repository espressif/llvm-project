// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### -no-canonical-prefixes -target xtensa \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "xtensa"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target xtensa --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-BAREMETAL %s

// C-XTENSA-BAREMETAL: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0/no-rtti"
// C-XTENSA-BAREMETAL: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib/no-rtti"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target xtensa --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree -frtti 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-BAREMETAL-RTTI %s

// C-XTENSA-BAREMETAL-RTTI: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-BAREMETAL-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0"
// C-XTENSA-BAREMETAL-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target xtensa --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree -mfix-esp32-psram-cache-issue 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-BAREMETAL-PSRAM %s

// C-XTENSA-BAREMETAL-PSRAM: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-BAREMETAL-PSRAM: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0/esp32-psram/no-rtti"
// C-XTENSA-BAREMETAL-PSRAM: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib/esp32-psram/no-rtti"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target xtensa --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_xtensa_tree -mfix-esp32-psram-cache-issue -frtti 2>&1 \
// RUN:   | FileCheck -check-prefix=C-XTENSA-BAREMETAL-PSRAM-RTTI %s

// C-XTENSA-BAREMETAL-PSRAM-RTTI: "{{.*}}Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}xtensa-esp32-elf-ld"
// C-XTENSA-BAREMETAL-PSRAM-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}8.4.0/esp32-psram"
// C-XTENSA-BAREMETAL-PSRAM-RTTI: "-L{{.*}}/Inputs/multilib_xtensa_tree/lib/gcc/xtensa-esp32-elf/8.4.0/../../..{{/|\\\\}}..{{/|\\\\}}xtensa-esp32-elf{{/|\\\\}}lib/esp32-psram"
