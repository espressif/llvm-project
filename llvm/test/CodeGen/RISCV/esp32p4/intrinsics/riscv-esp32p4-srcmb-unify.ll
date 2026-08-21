; RUN: llc -O2 -mattr=+xespv2p1,+espv-lowering -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=ASM2P2

define dso_local ptr @test_srcmb_s16_q_qacc(ptr noundef %src_data, ptr noundef %src_qw, ptr noundef %dst_qu) local_unnamed_addr #0 {
; ASM-LABEL: test_srcmb_s16_q_qacc:
; ASM:       esp.srcmb.s16.q.qacc
; ASM2P2-LABEL: test_srcmb_s16_q_qacc:
; ASM2P2:       esp.srcmb.s16.q.qacc
entry:
  %vld1 = tail call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src_data, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %ev2 = extractvalue { <16 x i8>, ptr } %vld1, 1
  %bc1 = bitcast <16 x i8> %ev1 to <8 x i16>
  %vld2 = tail call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %ev2, i32 16)
  %ev3 = extractvalue { <16 x i8>, ptr } %vld2, 0
  %bc2 = bitcast <16 x i8> %ev3 to <8 x i16>
  %v1 = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.riscv.esp.zero.qacc()
  %ev4 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v1, 0
  %ev5 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v1, 1
  %ev6 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v1, 2
  %ev7 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v1, 3
  %v2 = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.riscv.esp.vsmulas.s16.qacc(<16 x i8> %ev4, <16 x i8> %ev5, <16 x i8> %ev6, <16 x i8> %ev7, <8 x i16> %bc1, <8 x i16> %bc2, i32 1, i32 0)
  %ev8 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v2, 0
  %ev9 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v2, 1
  %ev10 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v2, 2
  %ev11 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v2, 3
  %vld3 = tail call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src_qw, i32 16)
  %ev12 = extractvalue { <16 x i8>, ptr } %vld3, 0
  %bc3 = bitcast <16 x i8> %ev12 to <8 x i16>
  %v3 = tail call <8 x i16> @llvm.riscv.esp.srcmb.s16.q.qacc(<16 x i8> %ev8, <16 x i8> %ev9, <16 x i8> %ev10, <16 x i8> %ev11, <8 x i16> %bc3, i32 1, i32 7)
  %bc4 = bitcast <8 x i16> %v3 to <16 x i8>
  %vst_ptr = tail call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %bc4, ptr %dst_qu, i32 16)
  ret ptr %vst_ptr
}

define dso_local ptr @test_srcmb_u16_qacc(ptr noundef %src, ptr noundef %dst) local_unnamed_addr #0 {
; ASM-LABEL: test_srcmb_u16_qacc:
; ASM:       esp.srcmb.u16.qacc
; ASM2P2-LABEL: test_srcmb_u16_qacc:
; ASM2P2:       esp.srcmb.u16.qacc
entry:
  %vld1 = tail call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src, i32 16)
  %ev1 = extractvalue { <16 x i8>, ptr } %vld1, 0
  %bc1 = bitcast <16 x i8> %ev1 to <8 x i16>
  %v1 = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.riscv.esp.zero.qacc()
  %ev2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v1, 0
  %ev3 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v1, 1
  %ev4 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v1, 2
  %ev5 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v1, 3
  %v2 = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.riscv.esp.vsmulas.s16.qacc(<16 x i8> %ev2, <16 x i8> %ev3, <16 x i8> %ev4, <16 x i8> %ev5, <8 x i16> %bc1, <8 x i16> %bc1, i32 1, i32 0)
  %ev6 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v2, 0
  %ev7 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v2, 1
  %ev8 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v2, 2
  %ev9 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %v2, 3
  %v3 = tail call <8 x i16> @llvm.riscv.esp.srcmb.u16.qacc(<16 x i8> %ev6, <16 x i8> %ev7, <16 x i8> %ev8, <16 x i8> %ev9, i32 0, i32 1, i32 0, i32 7)
  %bc2 = bitcast <8 x i16> %v3 to <16 x i8>
  %vst_ptr = tail call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %bc2, ptr %dst, i32 16)
  ret ptr %vst_ptr
}

declare { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr, i32)
declare ptr @llvm.riscv.esp.vst.128.ip(<16 x i8>, ptr, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.riscv.esp.zero.qacc()
declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.riscv.esp.vsmulas.s16.qacc(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <8 x i16>, <8 x i16>, i32, i32)

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+d,+f,+m,+relax,+v,+xandes,+xesp,+xesplower,+xespv,+zbb,+zbs,+zca,+zcb,+zcmp,+zfa,+zfh,+zfhmin,+zicsr,+zifencei,+zilsd,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvfh,+zvl128b,+zvl32b,+zvl64b,-b,-e,-experimental-smmpm,-experimental-ssamoswap-add,-experimental-ssfcefmin,-experimental-ssstrict,-experimental-sssync,-experimental-zacas,-experimental-zicfilp,-experimental-zicfiss,-experimental-ztso,-h,-save-restore,-sha,-shcounterenov,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcsrind,-smepmp,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscpmp,-sscsrind,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-sv32,-sv57,-sv64,-svinval,-svnapot,-svpbmt,-svvptc,-xandesbfhcvt,-xandesbfhcvtbf2i,-xandesbfhcvti2bf,-xandesperf,-xandesvbfhcvt,-xandesvbfhcvtbf2i,-xandesvbfhcvti2bf,-xandesvdot,-xandesvpack,-xandesvsll,-xespefhw,-xesplower,-xespv,-xespv2p1,-xespv2p2,-xsfcease,-xsfmm,-xsfmmbase,-xsfmmcmp,-xsfmmcp,-xsfmmf8,-xsfmmfcvt,-xsfmmim,-xsfmmint8,-xsfmmmode,-xsfmmop-s,-xsfmmop-t,-xsfmmops,-xsfmmopt,-xsfmmq,-xsfmmq2,-xsfmmqfcvt,-xsfmmqqq8,-xsfvcp,-xsfvfnrclip,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsha,-xshlcofideleg,-xshcounterenw,-xshgatpa,-xshlcofideleg,-xshlctvala,-xshlcofval,-xshsstvala,-xshsstvecd,-xshvctvala,-xshvctvecd,-xshvlctvala,-xshvlctvecd,-xshvsatpa,-xshvstvala,-xshvstvecd,-xsifivecdiscarddlone,-xsifivecflushdlone,-xssfr,-xssqosid,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zaamo,-zabha,-zalrsc,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zcd,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccif,-zicclsm,-ziccrse,-zicond,-zicond,-zihintntl,-zihintpause,-zihpm,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
