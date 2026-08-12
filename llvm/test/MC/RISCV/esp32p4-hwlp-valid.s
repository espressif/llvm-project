# RUN: llvm-mc %s -triple=riscv32 -mcpu=esp32p4 -show-encoding | FileCheck -check-prefixes=CHECK %s

dl_hwlp_test:
# CHECK: dl_hwlp_test:
    esp.lp.setup 0, a1, loop_last_instruction
# CHECK: esp.lp.setup	 0, a1, loop_last_instruction # encoding: [0x2b'A',0xc0'A',0x05'A',A]
# CHECK: #   fixup A - offset: 0, value: loop_last_instruction, kind: fixup_riscv_esp_lp_offset_12
    esp.lp.starti 0, loop_last_instruction
# CHECK: esp.lp.starti	 0, loop_last_instruction # encoding: [0x2b'A',A,A,A]
# CHECK: #   fixup A - offset: 0, value: loop_last_instruction, kind: fixup_riscv_esp_lp_offset_12
    esp.lp.counti 0, 4000
# CHECK: esp.lp.counti	 0, 4000                # encoding: [0x2b,0x30,0x00,0xfa]
    esp.lp.count 0, a1
# CHECK: esp.lp.count	 0, a1                  # encoding: [0x2b,0xa0,0x05,0x00]
    esp.lp.setupi 0, 1234, loop_last_instruction
# CHECK: esp.lp.setupi	 0, 1234, loop_last_instruction # encoding: [0x2b'A',0x50'A',0x20'A',0x4d'A']
# CHECK: #   fixup A - offset: 0, value: loop_last_instruction, kind: fixup_riscv_esp_lp_offset_9
    loop_last_instruction:
# CHECK: loop_last_instruction:
        addi a0, a0, 1
# CHECK: addi	a0, a0, 1                       # encoding: [0x05,0x05]
    ret
# CHECK: ret                                     # encoding: [0x82,0x80]

# Large constant loop offsets: esp.lp.setup/starti/endi take uimm13_step4
# ([0, 8190], even), so exercise offsets > 1022 (the setupi/uimm10_step4 max);
# esp.lp.setupi takes uimm10_step4 ([0, 1022], even). See fixup_riscv_esp_lp_offset_12/_9.
dl_hwlp_const_offsets:
# CHECK: dl_hwlp_const_offsets:
    esp.lp.setup 0, a1, 2048
# CHECK: esp.lp.setup	 0, a1, 2048            # encoding: [0x2b,0xc0,0x05,0x40]
    esp.lp.setup 0, a1, 8190
# CHECK: esp.lp.setup	 0, a1, 8190            # encoding: [0x2b,0xc0,0xf5,0xff]
    esp.lp.starti 1, 8190
# CHECK: esp.lp.starti	 1, 8190                # encoding: [0xab,0x00,0xf0,0xff]
    esp.lp.endi 0, 4094
# CHECK: esp.lp.endi	 0, 4094                # encoding: [0x2b,0x10,0xf0,0x7f]
    esp.lp.setupi 0, 100, 1022
# CHECK: esp.lp.setupi	 0, 100, 1022           # encoding: [0x2b,0xdf,0x4f,0x06]
