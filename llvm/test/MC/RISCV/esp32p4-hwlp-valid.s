# RUN: llvm-mc %s -triple=riscv32 -mcpu=esp32p4 -show-encoding | FileCheck -check-prefixes=CHECK %s

dl_hwlp_test:
# CHECK: dl_hwlp_test:
    esp.lp.setup 0, a1, loop_last_instruction
# CHECK: esp.lp.setup	 0, a1, loop_last_instruction # encoding: [0x2b'A',0xc0'A',0x05'A',A]
    esp.lp.starti 0, loop_last_instruction
# CHECK: esp.lp.starti	 0, loop_last_instruction # encoding: [0x2b'A',A,A,A]
    esp.lp.counti 0, 4000
# CHECK: esp.lp.counti	 0, 4000                # encoding: [0x2b,0x30,0x00,0xfa]
    esp.lp.count 0, a1
# CHECK: esp.lp.count	 0, a1                  # encoding: [0x2b,0xa0,0x05,0x00]
    esp.lp.setupi 0, 1234, loop_last_instruction
# CHECK: esp.lp.setupi	 0, 1234, loop_last_instruction # encoding: [0x2b'A',0x50'A',0x20'A',0x4d'A']
    # lp.setup 0, a1, loop_last_instruction
# CHECK: #   fixup A - offset: 0, value: loop_last_instruction, kind: fixup_riscv_branch
    loop_last_instruction:
# CHECK: loop_last_instruction:
        addi a0, a0, 1
# CHECK: addi	a0, a0, 1                       # encoding: [0x05,0x05]
    ret
# CHECK: ret                                     # encoding: [0x82,0x80]
