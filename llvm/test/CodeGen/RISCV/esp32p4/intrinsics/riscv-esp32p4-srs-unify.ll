; RUN: llc -O2 -mattr=+xespv2p1,+espv-lowering -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=ASM2P2

define dso_local i32 @test_srs_s_xacc(i32 noundef %xacc_h, i32 noundef %xacc_l, i32 noundef %shift) local_unnamed_addr #0 {
; ASM-LABEL: test_srs_s_xacc:
; ASM:       esp.srs.s.xacc
; ASM2P2-LABEL: test_srs_s_xacc:
; ASM2P2:       esp.srs.s.xacc
entry:
  %v1 = tail call { i32, i32, i32 } @llvm.riscv.esp.srs.s.xacc(i32 %xacc_h, i32 %xacc_l, i32 %shift, i32 0, i32 7)
  %ev0 = extractvalue { i32, i32, i32 } %v1, 0
  ret i32 %ev0
}

define dso_local i32 @test_srs_u_xacc(i32 noundef %xacc_h, i32 noundef %xacc_l, i32 noundef %shift) local_unnamed_addr #0 {
; ASM-LABEL: test_srs_u_xacc:
; ASM:       esp.srs.u.xacc
; ASM2P2-LABEL: test_srs_u_xacc:
; ASM2P2:       esp.srs.u.xacc
entry:
  %v1 = tail call { i32, i32, i32 } @llvm.riscv.esp.srs.u.xacc(i32 %xacc_h, i32 %xacc_l, i32 %shift, i32 0, i32 7)
  %ev0 = extractvalue { i32, i32, i32 } %v1, 0
  ret i32 %ev0
}

declare { i32, i32, i32 } @llvm.riscv.esp.srs.s.xacc(i32, i32, i32, i32 immarg, i32 immarg) #1
declare { i32, i32, i32 } @llvm.riscv.esp.srs.u.xacc(i32, i32, i32, i32 immarg, i32 immarg) #1

attributes #0 = { "target-features"="+32bit,+xespv2p1" }
attributes #1 = { nounwind }
