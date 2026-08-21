// REQUIRES: riscv-registered-target
//
// Verify that -march selects the correct xespv ISA version, enabling
// version-appropriate instruction syntax and encodings.

// RUN: %clang --target=riscv32-unknown-elf -march=rv32i_xespv2p1 -### -x assembler -c %S/Inputs/xespv-march-2p1.s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FEAT21
// RUN: %clang --target=riscv32-unknown-elf -march=rv32i_xespv2p2 -### -x assembler -c %S/Inputs/xespv-march-2p2.s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FEAT22
// RUN: %clang --target=riscv32-unknown-elf -march=rv32i_xespv -### -x assembler -c %S/Inputs/xespv-march-2p2.s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FEAT22

// FEAT21: "-target-feature" "+xespv2p1"
// FEAT22: "-target-feature" "+xespv"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32i_xespv2p1 -x assembler -c -o %t21.o %S/Inputs/xespv-march-2p1.s
// RUN: llvm-objdump -s --section=.text %t21.o | FileCheck %s --check-prefix=TEXT21
// RUN: %clang --target=riscv32-unknown-elf -march=rv32i_xespv2p2 -x assembler -c -o %t22.o %S/Inputs/xespv-march-2p2.s
// RUN: llvm-objdump -s --section=.text %t22.o | FileCheck %s --check-prefix=TEXT22
// RUN: %clang --target=riscv32-unknown-elf -march=rv32i_xespv -x assembler -c -o %t22.o %S/Inputs/xespv-march-2p2.s
// RUN: llvm-objdump -s --section=.text %t22.o | FileCheck %s --check-prefix=TEXT22

// TEXT21: 5f80874a
// TEXT22: 1ba086c2

// 2.2-only operand syntax must not assemble under xespv 2.1.
// RUN: not %clang --target=riscv32-unknown-elf -march=rv32i_xespv2p1 -x assembler -c -o %t.o %S/Inputs/xespv-march-2p2.s 2>&1

// The 2-operand form assembles under xespv 2.2 with a defaulted trunc (0)
// sat operand, producing the 2.2 encoding (distinct from TEXT21).
// RUN: %clang --target=riscv32-unknown-elf -march=rv32i_xespv2p2 -x assembler -c -o %t22b.o %S/Inputs/xespv-march-2p1.s
// RUN: llvm-objdump -s --section=.text %t22b.o | FileCheck %s --check-prefix=TEXT22B

// TEXT22B: 1ba0864a
