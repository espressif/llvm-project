; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 4
; RUN: opt -S -mtriple=riscv32-esp-unknown-elf -passes=riscv-loop-unroll-and-remainder -riscv-loop-unroll-and-remainder=true < %s | FileCheck %s
define dso_local noundef i32 @dsps_mulc_f32_ansi(ptr noalias noundef readonly %input, ptr noalias noundef writeonly %output, i32 noundef %len, float noundef %C, i32 noundef %step_in, i32 noundef %step_out) local_unnamed_addr {
; CHECK-LABEL: define dso_local noundef i32 @dsps_mulc_f32_ansi(
; CHECK-SAME: ptr noalias noundef readonly [[INPUT:%.*]], ptr noalias noundef writeonly [[OUTPUT:%.*]], i32 noundef [[LEN:%.*]], float noundef [[C:%.*]], i32 noundef [[STEP_IN:%.*]], i32 noundef [[STEP_OUT:%.*]]) local_unnamed_addr {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[INPUT]], null
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq ptr [[OUTPUT]], null
; CHECK-NEXT:    [[OR_COND:%.*]] = or i1 [[CMP]], [[CMP1]]
; CHECK-NEXT:    br i1 [[OR_COND]], label [[RETURN:%.*]], label [[IF_END:%.*]]
; CHECK:       if.end:
; CHECK-NEXT:    [[CMP4:%.*]] = icmp sgt i32 [[LEN]], 2
; CHECK-NEXT:    br i1 [[CMP4]], label [[FOR_COND_PREHEADER_NEW:%.*]], label [[FOR_COND_PREHEADER:%.*]]
; CHECK:       for.cond.preheader:
; CHECK-NEXT:    [[CMP413:%.*]] = icmp sgt i32 [[LEN]], 0
; CHECK-NEXT:    br i1 [[CMP413]], label [[FOR_BODY_CLONE:%.*]], label [[RETURN]]
; CHECK:       for.cond.preheader.new:
; CHECK-NEXT:    [[SUB:%.*]] = add nsw i32 [[LEN]], -16
; CHECK-NEXT:    [[CMP6_NOT207:%.*]] = icmp ult i32 [[LEN]], 16
; CHECK-NEXT:    br i1 [[CMP6_NOT207]], label [[FOR_COND_PREHEADER_NEW2:%.*]], label [[FOR_BODY_MODIFY:%.*]]
; CHECK:       for.cond.preheader.new2:
; CHECK-NEXT:    [[TMP0:%.*]] = phi i32 [ [[TMP1:%.*]], [[FOR_BODY_MODIFY]] ], [ 0, [[FOR_COND_PREHEADER_NEW]] ]
; CHECK-NEXT:    [[CMP85209:%.*]] = icmp slt i32 [[TMP0]], [[LEN]]
; CHECK-NEXT:    br i1 [[CMP85209]], label [[FOR_BODY:%.*]], label [[RETURN]]
; CHECK:       for.body.modify:
; CHECK-NEXT:    [[I_014_MODIFY:%.*]] = phi i32 [ [[TMP1]], [[FOR_BODY_MODIFY]] ], [ 0, [[FOR_COND_PREHEADER_NEW]] ]
; CHECK-NEXT:    [[TMP1]] = add nuw i32 [[I_014_MODIFY]], 16
; CHECK-NEXT:    [[ADD:%.*]] = or disjoint i32 [[I_014_MODIFY]], 1
; CHECK-NEXT:    [[ADD3:%.*]] = or disjoint i32 [[I_014_MODIFY]], 2
; CHECK-NEXT:    [[ADD6:%.*]] = or disjoint i32 [[I_014_MODIFY]], 3
; CHECK-NEXT:    [[ADD10:%.*]] = or disjoint i32 [[I_014_MODIFY]], 4
; CHECK-NEXT:    [[ADD13:%.*]] = or disjoint i32 [[I_014_MODIFY]], 5
; CHECK-NEXT:    [[ADD16:%.*]] = or disjoint i32 [[I_014_MODIFY]], 6
; CHECK-NEXT:    [[ADD19:%.*]] = or disjoint i32 [[I_014_MODIFY]], 7
; CHECK-NEXT:    [[ADD22:%.*]] = or disjoint i32 [[I_014_MODIFY]], 8
; CHECK-NEXT:    [[ADD25:%.*]] = or disjoint i32 [[I_014_MODIFY]], 9
; CHECK-NEXT:    [[ADD28:%.*]] = or disjoint i32 [[I_014_MODIFY]], 10
; CHECK-NEXT:    [[ADD31:%.*]] = or disjoint i32 [[I_014_MODIFY]], 11
; CHECK-NEXT:    [[ADD34:%.*]] = or disjoint i32 [[I_014_MODIFY]], 12
; CHECK-NEXT:    [[ADD37:%.*]] = or disjoint i32 [[I_014_MODIFY]], 13
; CHECK-NEXT:    [[ADD40:%.*]] = or disjoint i32 [[I_014_MODIFY]], 14
; CHECK-NEXT:    [[ADD43:%.*]] = or disjoint i32 [[I_014_MODIFY]], 15
; CHECK-NEXT:    [[ARRAYIDX_MODIFY:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[I_014_MODIFY]]
; CHECK-NEXT:    [[ARRAYIDX7_MODIFY:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[I_014_MODIFY]]
; CHECK-NEXT:    [[ARRAYIDX1:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD]]
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD]]
; CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD3]]
; CHECK-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD3]]
; CHECK-NEXT:    [[ARRAYIDX8:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD6]]
; CHECK-NEXT:    [[ARRAYIDX9:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD6]]
; CHECK-NEXT:    [[ARRAYIDX11:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD10]]
; CHECK-NEXT:    [[ARRAYIDX12:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD10]]
; CHECK-NEXT:    [[ARRAYIDX14:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD13]]
; CHECK-NEXT:    [[ARRAYIDX15:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD13]]
; CHECK-NEXT:    [[ARRAYIDX17:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD16]]
; CHECK-NEXT:    [[ARRAYIDX18:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD16]]
; CHECK-NEXT:    [[ARRAYIDX20:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD19]]
; CHECK-NEXT:    [[ARRAYIDX21:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD19]]
; CHECK-NEXT:    [[ARRAYIDX23:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD22]]
; CHECK-NEXT:    [[ARRAYIDX24:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD22]]
; CHECK-NEXT:    [[ARRAYIDX26:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD25]]
; CHECK-NEXT:    [[ARRAYIDX27:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD25]]
; CHECK-NEXT:    [[ARRAYIDX29:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD28]]
; CHECK-NEXT:    [[ARRAYIDX30:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD28]]
; CHECK-NEXT:    [[ARRAYIDX32:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD31]]
; CHECK-NEXT:    [[ARRAYIDX33:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD31]]
; CHECK-NEXT:    [[ARRAYIDX35:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD34]]
; CHECK-NEXT:    [[ARRAYIDX36:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD34]]
; CHECK-NEXT:    [[ARRAYIDX38:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD37]]
; CHECK-NEXT:    [[ARRAYIDX39:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD37]]
; CHECK-NEXT:    [[ARRAYIDX41:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD40]]
; CHECK-NEXT:    [[ARRAYIDX42:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD40]]
; CHECK-NEXT:    [[ARRAYIDX44:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[ADD43]]
; CHECK-NEXT:    [[ARRAYIDX45:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[ADD43]]
; CHECK-NEXT:    [[TMP2:%.*]] = load float, ptr [[ARRAYIDX_MODIFY]], align 4
; CHECK-NEXT:    [[TMP3:%.*]] = load float, ptr [[ARRAYIDX1]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = load float, ptr [[ARRAYIDX4]], align 4
; CHECK-NEXT:    [[TMP5:%.*]] = load float, ptr [[ARRAYIDX8]], align 4
; CHECK-NEXT:    [[TMP6:%.*]] = load float, ptr [[ARRAYIDX11]], align 4
; CHECK-NEXT:    [[TMP7:%.*]] = load float, ptr [[ARRAYIDX14]], align 4
; CHECK-NEXT:    [[TMP8:%.*]] = load float, ptr [[ARRAYIDX17]], align 4
; CHECK-NEXT:    [[TMP9:%.*]] = load float, ptr [[ARRAYIDX20]], align 4
; CHECK-NEXT:    [[TMP10:%.*]] = load float, ptr [[ARRAYIDX23]], align 4
; CHECK-NEXT:    [[TMP11:%.*]] = load float, ptr [[ARRAYIDX26]], align 4
; CHECK-NEXT:    [[TMP12:%.*]] = load float, ptr [[ARRAYIDX29]], align 4
; CHECK-NEXT:    [[TMP13:%.*]] = load float, ptr [[ARRAYIDX32]], align 4
; CHECK-NEXT:    [[TMP14:%.*]] = load float, ptr [[ARRAYIDX35]], align 4
; CHECK-NEXT:    [[TMP15:%.*]] = load float, ptr [[ARRAYIDX38]], align 4
; CHECK-NEXT:    [[TMP16:%.*]] = load float, ptr [[ARRAYIDX41]], align 4
; CHECK-NEXT:    [[TMP17:%.*]] = load float, ptr [[ARRAYIDX44]], align 4
; CHECK-NEXT:    [[MUL5_MODIFY:%.*]] = fmul float [[C]], [[TMP2]]
; CHECK-NEXT:    [[TMP18:%.*]] = fmul float [[C]], [[TMP3]]
; CHECK-NEXT:    [[TMP19:%.*]] = fmul float [[C]], [[TMP4]]
; CHECK-NEXT:    [[TMP20:%.*]] = fmul float [[C]], [[TMP5]]
; CHECK-NEXT:    [[TMP21:%.*]] = fmul float [[C]], [[TMP6]]
; CHECK-NEXT:    [[TMP22:%.*]] = fmul float [[C]], [[TMP7]]
; CHECK-NEXT:    [[TMP23:%.*]] = fmul float [[C]], [[TMP8]]
; CHECK-NEXT:    [[TMP24:%.*]] = fmul float [[C]], [[TMP9]]
; CHECK-NEXT:    [[TMP25:%.*]] = fmul float [[C]], [[TMP10]]
; CHECK-NEXT:    [[TMP26:%.*]] = fmul float [[C]], [[TMP11]]
; CHECK-NEXT:    [[TMP27:%.*]] = fmul float [[C]], [[TMP12]]
; CHECK-NEXT:    [[TMP28:%.*]] = fmul float [[C]], [[TMP13]]
; CHECK-NEXT:    [[TMP29:%.*]] = fmul float [[C]], [[TMP14]]
; CHECK-NEXT:    [[TMP30:%.*]] = fmul float [[C]], [[TMP15]]
; CHECK-NEXT:    [[TMP31:%.*]] = fmul float [[C]], [[TMP16]]
; CHECK-NEXT:    [[TMP32:%.*]] = fmul float [[C]], [[TMP17]]
; CHECK-NEXT:    store float [[MUL5_MODIFY]], ptr [[ARRAYIDX7_MODIFY]], align 4
; CHECK-NEXT:    store float [[TMP18]], ptr [[ARRAYIDX2]], align 4
; CHECK-NEXT:    store float [[TMP19]], ptr [[ARRAYIDX5]], align 4
; CHECK-NEXT:    store float [[TMP20]], ptr [[ARRAYIDX9]], align 4
; CHECK-NEXT:    store float [[TMP21]], ptr [[ARRAYIDX12]], align 4
; CHECK-NEXT:    store float [[TMP22]], ptr [[ARRAYIDX15]], align 4
; CHECK-NEXT:    store float [[TMP23]], ptr [[ARRAYIDX18]], align 4
; CHECK-NEXT:    store float [[TMP24]], ptr [[ARRAYIDX21]], align 4
; CHECK-NEXT:    store float [[TMP25]], ptr [[ARRAYIDX24]], align 4
; CHECK-NEXT:    store float [[TMP26]], ptr [[ARRAYIDX27]], align 4
; CHECK-NEXT:    store float [[TMP27]], ptr [[ARRAYIDX30]], align 4
; CHECK-NEXT:    store float [[TMP28]], ptr [[ARRAYIDX33]], align 4
; CHECK-NEXT:    store float [[TMP29]], ptr [[ARRAYIDX36]], align 4
; CHECK-NEXT:    store float [[TMP30]], ptr [[ARRAYIDX39]], align 4
; CHECK-NEXT:    store float [[TMP31]], ptr [[ARRAYIDX42]], align 4
; CHECK-NEXT:    store float [[TMP32]], ptr [[ARRAYIDX45]], align 4
; CHECK-NEXT:    [[EXITCOND_NOT_MODIFY:%.*]] = icmp sgt i32 [[TMP1]], [[SUB]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT_MODIFY]], label [[FOR_COND_PREHEADER_NEW2]], label [[FOR_BODY_MODIFY]]
; CHECK:       for.body:
; CHECK-NEXT:    [[I_014:%.*]] = phi i32 [ [[INC:%.*]], [[FOR_BODY]] ], [ [[TMP0]], [[FOR_COND_PREHEADER_NEW2]] ]
; CHECK-NEXT:    [[MUL:%.*]] = mul nsw i32 [[I_014]], [[STEP_IN]]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[MUL]]
; CHECK-NEXT:    [[TMP33:%.*]] = load float, ptr [[ARRAYIDX]], align 4
; CHECK-NEXT:    [[MUL5:%.*]] = fmul float [[C]], [[TMP33]]
; CHECK-NEXT:    [[MUL6:%.*]] = mul nsw i32 [[I_014]], [[STEP_OUT]]
; CHECK-NEXT:    [[ARRAYIDX7:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[MUL6]]
; CHECK-NEXT:    store float [[MUL5]], ptr [[ARRAYIDX7]], align 4
; CHECK-NEXT:    [[INC]] = add nuw nsw i32 [[I_014]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i32 [[INC]], [[LEN]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[RETURN]], label [[FOR_BODY]]
; CHECK:       for.body.clone:
; CHECK-NEXT:    [[I_014_CLONE:%.*]] = phi i32 [ [[INC_CLONE:%.*]], [[FOR_BODY_CLONE]] ], [ 0, [[FOR_COND_PREHEADER]] ]
; CHECK-NEXT:    [[MUL_CLONE:%.*]] = mul nsw i32 [[I_014_CLONE]], [[STEP_IN]]
; CHECK-NEXT:    [[ARRAYIDX_CLONE:%.*]] = getelementptr inbounds float, ptr [[INPUT]], i32 [[MUL_CLONE]]
; CHECK-NEXT:    [[TMP34:%.*]] = load float, ptr [[ARRAYIDX_CLONE]], align 4
; CHECK-NEXT:    [[MUL5_CLONE:%.*]] = fmul float [[C]], [[TMP34]]
; CHECK-NEXT:    [[MUL6_CLONE:%.*]] = mul nsw i32 [[I_014_CLONE]], [[STEP_OUT]]
; CHECK-NEXT:    [[ARRAYIDX7_CLONE:%.*]] = getelementptr inbounds float, ptr [[OUTPUT]], i32 [[MUL6_CLONE]]
; CHECK-NEXT:    store float [[MUL5_CLONE]], ptr [[ARRAYIDX7_CLONE]], align 4
; CHECK-NEXT:    [[INC_CLONE]] = add nuw nsw i32 [[I_014_CLONE]], 1
; CHECK-NEXT:    [[EXITCOND_NOT_CLONE:%.*]] = icmp eq i32 [[INC_CLONE]], [[LEN]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT_CLONE]], label [[RETURN]], label [[FOR_BODY_CLONE]]
; CHECK:       return:
; CHECK-NEXT:    [[RETVAL_0:%.*]] = phi i32 [ 458755, [[ENTRY:%.*]] ], [ 0, [[FOR_COND_PREHEADER]] ], [ 0, [[FOR_BODY]] ], [ 0, [[FOR_BODY_CLONE]] ], [ 0, [[FOR_COND_PREHEADER_NEW2]] ]
; CHECK-NEXT:    ret i32 [[RETVAL_0]]
;
entry:
  %cmp = icmp eq ptr %input, null
  %cmp1 = icmp eq ptr %output, null
  %or.cond = or i1 %cmp, %cmp1
  br i1 %or.cond, label %return, label %if.end

if.end:                                           ; preds = %entry
  %cmp4 = icmp sgt i32 %len, 2
  br i1 %cmp4, label %for.body, label %for.cond.preheader

for.cond.preheader:                               ; preds = %if.end
  %cmp413 = icmp sgt i32 %len, 0
  br i1 %cmp413, label %for.body.clone, label %return

for.body:                                         ; preds = %for.body, %if.end
  %i.014 = phi i32 [ %inc, %for.body ], [ 0, %if.end ]
  %mul = mul nsw i32 %i.014, %step_in
  %arrayidx = getelementptr inbounds float, ptr %input, i32 %mul
  %0 = load float, ptr %arrayidx, align 4
  %mul5 = fmul float %0, %C
  %mul6 = mul nsw i32 %i.014, %step_out
  %arrayidx7 = getelementptr inbounds float, ptr %output, i32 %mul6
  store float %mul5, ptr %arrayidx7, align 4
  %inc = add nuw nsw i32 %i.014, 1
  %exitcond.not = icmp eq i32 %inc, %len
  br i1 %exitcond.not, label %return, label %for.body

for.body.clone:                                   ; preds = %for.body.clone, %for.cond.preheader
  %i.014.clone = phi i32 [ %inc.clone, %for.body.clone ], [ 0, %for.cond.preheader ]
  %mul.clone = mul nsw i32 %i.014.clone, %step_in
  %arrayidx.clone = getelementptr inbounds float, ptr %input, i32 %mul.clone
  %1 = load float, ptr %arrayidx.clone, align 4
  %mul5.clone = fmul float %1, %C
  %mul6.clone = mul nsw i32 %i.014.clone, %step_out
  %arrayidx7.clone = getelementptr inbounds float, ptr %output, i32 %mul6.clone
  store float %mul5.clone, ptr %arrayidx7.clone, align 4
  %inc.clone = add nuw nsw i32 %i.014.clone, 1
  %exitcond.not.clone = icmp eq i32 %inc.clone, %len
  br i1 %exitcond.not.clone, label %return, label %for.body.clone

return:                                           ; preds = %for.body.clone, %for.body, %for.cond.preheader, %entry
  %retval.0 = phi i32 [ 458755, %entry ], [ 0, %for.cond.preheader ], [ 0, %for.body ], [ 0, %for.body.clone ]
  ret i32 %retval.0
}
