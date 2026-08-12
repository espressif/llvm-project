# RUN: not llvm-mc %s -triple=riscv32 -mcpu=esp32p4 2>&1 | FileCheck %s

# esp.lp.setup/starti/endi take uimm13_step4: an even byte offset in [0, 8190].
# Out-of-range and odd constants must produce a clean diagnostic (previously an
# out-of-range constant hit llvm_unreachable and crashed the assembler, and odd
# constants were silently accepted and only caught by an encoder assert).

esp.lp.setup 0, a1, 8192
# CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 8190]
esp.lp.setup 0, a1, 2049
# CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 8190]
esp.lp.starti 0, 4097
# CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 8190]
esp.lp.endi 0, 8191
# CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 8190]

# esp.lp.setupi takes uimm10_step4: an even byte offset in [0, 1022].
esp.lp.setupi 0, 100, 1024
# CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 1022]
esp.lp.setupi 0, 100, 1021
# CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 1022]
