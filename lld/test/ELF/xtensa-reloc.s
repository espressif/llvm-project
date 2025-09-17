# REQUIRES: xtensa
# RUN: llvm-mc -filetype=obj -triple=xtensa -mcpu=esp32 %s -o %t.o
# RUN: ld.lld %t.o --defsym=a=0x2000 --section-start=.CALL=0x1000 --defsym=b=40 -o %t
# RUN: llvm-objdump -d --print-imm-hex %t | FileCheck %s

.section .CALL,"ax",@progbits
# CHECK-LABEL: section .CALL:
# CHECK:      call0 . +4096
# CHECK-NEXT: call0 . +4096
# CHECK-NEXT: call0 . +4092
# CHECK-NEXT: call0 . +4088
# CHECK-NEXT: call0 . -4068
  call0 a
  call0 a
  call0 a
  call0 a
  call0 b
