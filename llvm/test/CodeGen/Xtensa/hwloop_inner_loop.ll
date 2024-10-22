; RUN: llc -O1 -mtriple=xtensa -mcpu=esp32 -disable-xtensa-hwloops=false %s -o - | FileCheck %s


; Function Attrs: norecurse nounwind optsize readnone
define i32 @test_hwloop(i32 %a, i32 %b, i32 %n) local_unnamed_addr #0 {
; CHECK-LABEL: @test_hwloop
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %a.addr.0.lcssa = phi i32 [ %a, %entry ], [ %a.addr.1, %for.body ]
  ret i32 %a.addr.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %a.addr.08 = phi i32 [ %a.addr.1, %for.body ], [ %a, %entry ]
  %cmp1 = icmp sgt i32 %a.addr.08, 0
  %mul = mul nsw i32 %a.addr.08, %b
  %add = select i1 %cmp1, i32 %mul, i32 0
  %a.addr.1 = add nsw i32 %add, %a.addr.08
  %inc = add nuw nsw i32 %i.09, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.cond.cleanup
; CHECK: [[FOR_BODY:.LBB[0-9_]+]]:
; CHECK: loop  a4, [[LOOP_END:.LBB[0-9_]+]]
; CHECK: bge   a8, a2, [[FOR_BODY]]
; CHECK: nop
; CHECK-NEXT: [[LOOP_END]]:
; CHECK: j .LBB
}

