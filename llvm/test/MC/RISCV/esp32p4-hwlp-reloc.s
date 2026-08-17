# RUN: llvm-mc -triple=riscv32 -mattr=+xesploop -filetype=obj %s \
# RUN:   | llvm-readobj -r - | FileCheck %s

# A hardware-loop body is contiguous code, so its offset is resolved and
# pre-filled at assembly time; it also carries a relocation so the linker can
# re-scatter the offset if the body is relaxed (matching the GNU assembler).
# Each R_RISCV_ESP_LP_OFFSET_* must be immediately preceded by an
# R_RISCV_VENDOR relocation against a local symbol named "esp" (the Espressif
# binutils fork requires the pair and resolves them as
# R_RISCV_ESP_LP_OFFSET_9/12).

foo:
	esp.lp.setupi 0, 4095, lbl
	esp.lp.starti 0, lbl
	nop
lbl:

# CHECK:      Relocations [
# CHECK:        0x0 R_RISCV_VENDOR esp 0x0
# CHECK-NEXT:   0x0 R_RISCV_ESP_LP_OFFSET_9 lbl 0x0
# CHECK-NEXT:   0x4 R_RISCV_VENDOR esp 0x0
# CHECK-NEXT:   0x4 R_RISCV_ESP_LP_OFFSET_12 lbl 0x0
# CHECK:      ]
