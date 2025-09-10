; RUN: not llc -mtriple=xtensa -mcpu=generic < %s 2>&1 | FileCheck %s

define void @constraint_f() nounwind {
; CHECK: error: could not allocate input reg for constraint 'f'
  tail call void asm "add.s f0, f1, $0", "f"(float 0.0)
  ret void
}

define i32 @register_a100(i32 %a) nounwind {
; CHECK: error: could not allocate input reg for constraint '{$a100}'
  %1 = tail call i32 asm "addi $0, $1, 1", "=r,{$a100}"(i32 %a)
  ret i32 %1
}
