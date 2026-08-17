# RUN: llvm-mc -triple=riscv32 -mattr=+xesploop,+relax -filetype=obj %s \
# RUN:   | llvm-readobj -r - | FileCheck %s

# A hardware-loop body may contain an instruction that the assembler can
# relax, e.g. a conditional branch sitting between the setup and its target.
# The target is still in the same section: keep the assembly-time offset in the
# instruction (computed from the final layout) and emit the relocation so the
# linker can re-scatter it if the body size changes, matching the GNU
# assembler. Each R_RISCV_ESP_LP_OFFSET_* must be immediately preceded by an
# R_RISCV_VENDOR relocation against a local symbol named "esp" (the Espressif
# binutils fork requires the pair and resolves it as R_RISCV_ESP_LP_OFFSET_9/12).

bar:
	esp.lp.setupi 0, 4095, bar_lbl
	blt a1, a2, .bar_tgt
	nop
.bar_tgt:
bar_lbl:
	nop

# CHECK:      Relocations [
# CHECK:        0x0 R_RISCV_VENDOR esp 0x0
# CHECK-NEXT:   0x0 R_RISCV_ESP_LP_OFFSET_9 bar_lbl 0x0
# CHECK-NEXT:   0x4 R_RISCV_BRANCH .bar_tgt 0x0
# CHECK:      ]
