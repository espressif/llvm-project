# RUN: llvm-mc -triple=riscv32 -mattr=+xesploop -filetype=obj %s \
# RUN:   | llvm-readobj -r - | FileCheck %s

# A hardware-loop body is contiguous code, so its offset is resolved and
# pre-filled at assembly time; it also carries a relocation so the linker can
# re-scatter the offset if the body is relaxed (matching the GNU assembler).
#
# The Espressif relocation numbers R_RISCV_ESP_LP_OFFSET_9 (62) and
# R_RISCV_ESP_LP_OFFSET_12 (63) overlap upstream's R_RISCV_TLSDESC_HI20 /
# R_RISCV_TLSDESC_LOAD_LO12 at the same numbers, so llvm-readobj (which only
# knows the upstream names) prints those; the Espressif linker resolves them
# correctly.

foo:
	esp.lp.setupi 0, 4095, lbl
	esp.lp.starti 0, lbl
	nop
lbl:

# CHECK:      Relocations [
# CHECK:        0x0 R_RISCV_TLSDESC_HI20 lbl 0x0
# CHECK-NEXT:   0x4 R_RISCV_TLSDESC_LOAD_LO12 lbl 0x0
# CHECK:      ]
