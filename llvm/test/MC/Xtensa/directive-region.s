# RUN: llvm-mc -triple xtensa-esp-elf -filetype obj -o - %s \
# RUN:   | llvm-readobj -S --sd - \
# RUN:   | FileCheck %s

	.text
    .begin literal_prefix .ExceptionVector
	.literal_position
	.literal .LCPI0_0, 305419896
	.global	test_literal
	.p2align	2
	.type	test_literal,@function
test_literal:
# %bb.0:
	entry	a1, 32
	mov.n	a7, a1
	l32r	a2, .LCPI0_0
	retw.n
    .end literal_prefix

# CHECK: Section {
# CHECK:   Name: .ExceptionVector.literal
# CHECK:   SectionData (
# CHECK:     0000: 78563412
# CHECK:   )
# CHECK: }