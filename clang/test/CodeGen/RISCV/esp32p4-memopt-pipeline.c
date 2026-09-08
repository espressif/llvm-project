// Verify -mllvm -riscv-esp32p4-memopt wires memmove+memcpy (mem-intrin) at -O3 only,
// and does not double-schedule when individual flags are also set.

// RUN: %clang --target=riscv32-esp-unknown-elf -O3 -c -o /dev/null \
// RUN:   -mllvm -riscv-esp32p4-memopt -mllvm -print-pipeline-passes %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ON
// RUN: %clang --target=riscv32-esp-unknown-elf -O3 -c -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=OFF
// RUN: %clang --target=riscv32-esp-unknown-elf -O2 -c -o /dev/null \
// RUN:   -mllvm -riscv-esp32p4-memopt -mllvm -print-pipeline-passes %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=OFF
// RUN: %clang --target=riscv32-esp-unknown-elf -O3 -c -o /dev/null \
// RUN:   -mllvm -riscv-esp32p4-memopt \
// RUN:   -mllvm -riscv-esp32-p4-memmove -mllvm -riscv-esp32-p4-mem-intrin \
// RUN:   -mllvm -print-pipeline-passes %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ONCE

// ON: function(RISCVESP32P4MemmovePass,RISCVEsp32P4MemIntrinPass)
// OFF-NOT: RISCVESP32P4MemmovePass
// OFF-NOT: RISCVEsp32P4MemIntrinPass
// ONCE-COUNT-1: RISCVESP32P4MemmovePass
// ONCE-COUNT-1: RISCVEsp32P4MemIntrinPass

void foo(void) {}
