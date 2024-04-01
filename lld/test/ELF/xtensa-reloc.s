# REQUIRES: xtensa
# RUN: llvm-mc -filetype=obj -triple=xtensa -mcpu=esp32 %s -o %t.o
# RUN: ld.lld %t.o --defsym=a=0x2000 --section-start=.CALL=0x1000 --defsym=b=0x40 --defsym=c=0x140 --section-start=.BRANCH=0x5000 --defsym=d=0x5010 --section-start=.BR12=0x100 -o %t
# RUN: llvm-objdump -d --print-imm-hex %t | FileCheck %s

.section .BR12,"ax",@progbits
# CHECK-LABEL:section .BR12
# CHECK:      beqz a2, . +64
# CHECK-NEXT: bnez a3, . +61
# CHECK-NEXT: bgez a4, . +58
# CHECK-NEXT: bltz a5, . +55
  beqz a2, c
  bnez a3, c
  bgez a4, c
  bltz a5, c

.section .CALL,"ax",@progbits
# CHECK-LABEL: section .CALL:
# CHECK:      call0 . +4096
# CHECK-NEXT: call0 . +4096
# CHECK-NEXT: call0 . +4092
# CHECK-NEXT: call0 . +4088
# CHECK-NEXT: j     . +4084
# CHECK-NEXT: j     . +4081
# CHECK-NEXT: j     . +4078
# CHECK-NEXT: j     . -4053
# CHECK-NEXT: j     . -3800
# CHECK-NEXT: call0 . -4056
# CHECK-NEXT: call0 . -3804
# CHECK-NEXT: l32r a3, . -4065
# CHECK-NEXT: callx0 a3
# CHECK-NEXT: l32r a4, . -3815
# CHECK-NEXT: callx0 a4
  call0 a
  call0 a
  call0 a
  call0 a
  j a
  j a
  j a
  j b
  j c
  call0 b
  call0 c
  l32r a3, b
  callx0 a3
  l32r a4, c
  callx0 a4

.section .BRANCH,"ax",@progbits
# CHECK-LABEL: section .BRANCH:
# CHECK:      beq a3, a4, . +16
# CHECK-NEXT: ball a3, a4, . +13
# CHECK-NEXT: blt a3, a4, . +10
  beq a3, a4, d
  ball a3, a4, d
  blt a3, a4, d
