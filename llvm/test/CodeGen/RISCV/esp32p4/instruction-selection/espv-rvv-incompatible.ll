; RUN: not llc -mtriple=riscv32 -mattr=+xespv2p1,+v %s -o /dev/null 2>&1 \
; RUN:   | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xespv,+v %s -o /dev/null 2>&1 \
; RUN:   | FileCheck %s

; CHECK: LLVM ERROR: ESPV does not support RVV/Zve extensions

define void @empty() {
  ret void
}
