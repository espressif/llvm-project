// RUN: %clang -target xtensa -Wno-missing-multilib -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-XTENSA %s
// RUN: %clang -target xtensa-esp-unknown-elf -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-XTENSA %s
// CHECK-XTENSA: #define __XTENSA__ 1
// CHECK-XTENSA: #define __xtensa__ 1

// RUN: %clang -target xtensa-esp-unknown-elf -mcpu=esp32 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-ESP %s
// RUN: %clang -target xtensa-esp-unknown-elf -mcpu=esp32s2 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-ESP %s
// RUN: %clang -target xtensa-esp-unknown-elf -mcpu=esp32s3 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-ESP %s
// CHECK-ESP: #define __XCHAL_DCACHE_IS_WRITEBACK 0
// CHECK-ESP: #define __XCHAL_DCACHE_LINESIZE 16
// CHECK-ESP: #define __XCHAL_DCACHE_LINEWIDTH 4
// CHECK-ESP: #define __XCHAL_DCACHE_SIZE 0
// CHECK-ESP: #define __XCHAL_DEBUGLEVEL 6
// CHECK-ESP: #define __XCHAL_HAVE_ABS 1
// CHECK-ESP: #define __XCHAL_HAVE_ADDX 1
// CHECK-ESP: #define __XCHAL_HAVE_BE 0
// CHECK-ESP: #define __XCHAL_HAVE_CONST16 0
// CHECK-ESP: #define __XCHAL_HAVE_DEBUG 1
// CHECK-ESP: #define __XCHAL_HAVE_DENSITY 1
// CHECK-ESP: #define __XCHAL_HAVE_DIV32 1
// CHECK-ESP: #define __XCHAL_HAVE_L32R 1
// CHECK-ESP: #define __XCHAL_HAVE_MINMAX 1
// CHECK-ESP: #define __XCHAL_HAVE_MMU 0
// CHECK-ESP: #define __XCHAL_HAVE_MUL16 1
// CHECK-ESP: #define __XCHAL_HAVE_MUL32 1
// CHECK-ESP: #define __XCHAL_HAVE_MUL32_HIGH 1
// CHECK-ESP: #define __XCHAL_HAVE_NSA 1
// CHECK-ESP: #define __XCHAL_HAVE_PREDICTED_BRANCHES 0
// CHECK-ESP: #define __XCHAL_HAVE_RELEASE_SYNC 1
// CHECK-ESP: #define __XCHAL_HAVE_SEXT 1
// CHECK-ESP: #define __XCHAL_HAVE_THREADPTR 1
// CHECK-ESP: #define __XCHAL_HAVE_WIDE_BRANCHES 0
// CHECK-ESP: #define __XCHAL_HAVE_WINDOWED 1
// CHECK-ESP: #define __XCHAL_ICACHE_LINESIZE 16
// CHECK-ESP: #define __XCHAL_ICACHE_LINEWIDTH 4
// CHECK-ESP: #define __XCHAL_ICACHE_SIZE 0
// CHECK-ESP: #define __XCHAL_INST_FETCH_WIDTH 4
// CHECK-ESP: #define __XCHAL_NUM_AREGS 64
// CHECK-ESP: #define __XCHAL_NUM_DBREAK 2
// CHECK-ESP: #define __XCHAL_NUM_IBREAK 2
// CHECK-ESP: #define __XTENSA_EL__ 1
// CHECK-ESP: #define __XTENSA_WINDOWED_ABI__ 1

// RUN: %clang -target xtensa-esp-unknown-elf -mcpu=esp32 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-ESP32_S3 %s
// RUN: %clang -target xtensa-esp-unknown-elf -mcpu=esp32s3 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-ESP32_S3 %s
// CHECK-ESP32_S3: #define __XCHAL_HAVE_BOOLEANS 1
// CHECK-ESP32_S3: #define __XCHAL_HAVE_FP 1
// CHECK-ESP32_S3: #define __XCHAL_HAVE_FP_DIV 1
// CHECK-ESP32_S3: #define __XCHAL_HAVE_FP_POSTINC 1
// CHECK-ESP32_S3: #define __XCHAL_HAVE_FP_RECIP 1
// CHECK-ESP32_S3: #define __XCHAL_HAVE_FP_RSQRT 1
// CHECK-ESP32_S3: #define __XCHAL_HAVE_FP_SQRT 1
// CHECK-ESP32_S3: #define __XCHAL_HAVE_LOOPS 1
// CHECK-ESP32_S3: #define __XCHAL_HAVE_MAC16 1
// CHECK-ESP32_S3: #define __XCHAL_HAVE_S32C1I 1

// RUN: %clang -target xtensa-esp-unknown-elf -mcpu=esp32 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-ESP32 %s
// CHECK-ESP32: #define __XCHAL_HAVE_DFP_ACCEL 1
// CHECK-ESP32: #define __XCHAL_HAVE_DFP_DIV 1
// CHECK-ESP32: #define __XCHAL_HAVE_DFP_RECIP 1
// CHECK-ESP32: #define __XCHAL_HAVE_DFP_RSQRT 1
// CHECK-ESP32: #define __XCHAL_HAVE_DFP_SQRT 1
// CHECK-ESP32: #define __XCHAL_MAX_INSTRUCTION_SIZE 3

// RUN: %clang -target xtensa-esp-unknown-elf -mcpu=esp32s3 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-ESP32S3 %s
// CHECK-ESP32S3: #define __XCHAL_HAVE_DFP_ACCEL 0
// CHECK-ESP32S3: #define __XCHAL_HAVE_DFP_DIV 0
// CHECK-ESP32S3: #define __XCHAL_HAVE_DFP_RECIP 0
// CHECK-ESP32S3: #define __XCHAL_HAVE_DFP_RSQRT 0
// CHECK-ESP32S3: #define __XCHAL_HAVE_DFP_SQRT 0
// CHECK-ESP32S3: #define __XCHAL_MAX_INSTRUCTION_SIZE 4

// RUN: %clang -target xtensa-esp-unknown-elf -mcpu=esp32s2 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-ESP32S2 %s
// CHECK-ESP32S2: #define __XCHAL_HAVE_BOOLEANS 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_DFP_ACCEL 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_DFP_DIV 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_DFP_RECIP 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_DFP_RSQRT 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_DFP_SQRT 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_FP 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_FP_DIV 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_FP_POSTINC 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_FP_RECIP 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_FP_RSQRT 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_FP_SQRT 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_LOOPS 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_MAC16 0
// CHECK-ESP32S2: #define __XCHAL_HAVE_S32C1I 0
// CHECK-ESP32S2: #define __XCHAL_MAX_INSTRUCTION_SIZE 3
