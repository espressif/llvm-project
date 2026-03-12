# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-xtheadzmpanel -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENCODING %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-xtheadzmpanel %s \
# RUN:     | llvm-objdump --mattr=+experimental-xtheadzmpanel -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-DISASM %s

#===----------------------------------------------------------------------===#
# Configuration instructions
#===----------------------------------------------------------------------===#

# CHECK-ENCODING: encoding: [0x2b,0x20,0x05,0x00]
# CHECK-DISASM: th.mset22adra a0
th.mset22adra a0

# CHECK-ENCODING: encoding: [0x2b,0xa0,0x05,0x10]
# CHECK-DISASM: th.mset22adrb a1
th.mset22adrb a1

# CHECK-ENCODING: encoding: [0x2b,0x20,0x06,0x20]
# CHECK-DISASM: th.mset22adrd a2
th.mset22adrd a2

# CHECK-ENCODING: encoding: [0x2b,0xa0,0x06,0x30]
# CHECK-DISASM: th.mset22rsba a3
th.mset22rsba a3

# CHECK-ENCODING: encoding: [0x2b,0x20,0x07,0x40]
# CHECK-DISASM: th.mset22rsbb a4
th.mset22rsbb a4

# CHECK-ENCODING: encoding: [0x2b,0xa0,0x07,0x50]
# CHECK-DISASM: th.mset22rsbd a5
th.mset22rsbd a5

# CHECK-ENCODING: encoding: [0x2b,0xa0,0x02,0x60]
# CHECK-DISASM: th.mset22m t0
th.mset22m t0

# CHECK-ENCODING: encoding: [0x2b,0x20,0x03,0x70]
# CHECK-DISASM: th.mset22n t1
th.mset22n t1

# CHECK-ENCODING: encoding: [0x2b,0xa0,0x03,0x80]
# CHECK-DISASM: th.mset22k t2
th.mset22k t2

# CHECK-ENCODING: encoding: [0x2b,0x20,0x0e,0x90]
# CHECK-DISASM: th.msetrstptr t3
th.msetrstptr t3

# CHECK-ENCODING: encoding: [0x2b,0xa0,0x0e,0xa0]
# CHECK-DISASM: th.msetaccum t4
th.msetaccum t4

# CHECK-ENCODING: encoding: [0x2b,0x20,0x0f,0xb0]
# CHECK-DISASM: th.msetoob t5
th.msetoob t5

#===----------------------------------------------------------------------===#
# Configuration instructions with different registers
#===----------------------------------------------------------------------===#

# CHECK-ENCODING: encoding: [0x2b,0x20,0x00,0x00]
# CHECK-DISASM: th.mset22adra zero
th.mset22adra zero

# CHECK-ENCODING: encoding: [0x2b,0xa0,0x00,0x10]
# CHECK-DISASM: th.mset22adrb ra
th.mset22adrb ra

# CHECK-ENCODING: encoding: [0x2b,0x20,0x01,0x20]
# CHECK-DISASM: th.mset22adrd sp
th.mset22adrd sp

# CHECK-ENCODING: encoding: [0x2b,0xa0,0x0f,0x30]
# CHECK-DISASM: th.mset22rsba t6
th.mset22rsba t6

# CHECK-ENCODING: encoding: [0x2b,0x20,0x04,0x60]
# CHECK-DISASM: th.mset22m s0
th.mset22m s0

# CHECK-ENCODING: encoding: [0x2b,0x20,0x05,0x70]
# CHECK-DISASM: th.mset22n a0
th.mset22n a0

# CHECK-ENCODING: encoding: [0x2b,0xa0,0x05,0x80]
# CHECK-DISASM: th.mset22k a1
th.mset22k a1

# CHECK-ENCODING: encoding: [0x2b,0x20,0x06,0x90]
# CHECK-DISASM: th.msetrstptr a2
th.msetrstptr a2

# CHECK-ENCODING: encoding: [0x2b,0xa0,0x06,0xa0]
# CHECK-DISASM: th.msetaccum a3
th.msetaccum a3

# CHECK-ENCODING: encoding: [0x2b,0x20,0x07,0xb0]
# CHECK-DISASM: th.msetoob a4
th.msetoob a4

#===----------------------------------------------------------------------===#
# Load instructions
#===----------------------------------------------------------------------===#

# CHECK-ENCODING: encoding: [0x2b,0x20,0x00,0x04]
# CHECK-DISASM: th.ml22e8
th.ml22e8

# CHECK-ENCODING: encoding: [0x2b,0x24,0x00,0x04]
# CHECK-DISASM: th.ml22e16
th.ml22e16

#===----------------------------------------------------------------------===#
# Store instructions
#===----------------------------------------------------------------------===#

# CHECK-ENCODING: encoding: [0x2b,0x24,0x00,0x26]
# CHECK-DISASM: th.msc22e16
th.msc22e16

# CHECK-ENCODING: encoding: [0x2b,0x28,0x00,0x26]
# CHECK-DISASM: th.msc22e32
th.msc22e32

#===----------------------------------------------------------------------===#
# FP matmul instructions (external operand variants)
#===----------------------------------------------------------------------===#

# CHECK-ENCODING: encoding: [0x2b,0x24,0x00,0x08]
# CHECK-DISASM: th.mfmacc22.h.e5
th.mfmacc22.h.e5

# CHECK-ENCODING: encoding: [0x2b,0x24,0x80,0x08]
# CHECK-DISASM: th.mfmacc22.h.e4
th.mfmacc22.h.e4

# CHECK-ENCODING: encoding: [0x2b,0x24,0x00,0x0a]
# CHECK-DISASM: th.mfmacc22.bf16.e5
th.mfmacc22.bf16.e5

# CHECK-ENCODING: encoding: [0x2b,0x24,0x80,0x0a]
# CHECK-DISASM: th.mfmacc22.bf16.e4
th.mfmacc22.bf16.e4

# CHECK-ENCODING: encoding: [0x2b,0x28,0x00,0x08]
# CHECK-DISASM: th.mfmacc22.s.e5
th.mfmacc22.s.e5

# CHECK-ENCODING: encoding: [0x2b,0x28,0x80,0x08]
# CHECK-DISASM: th.mfmacc22.s.e4
th.mfmacc22.s.e4

#===----------------------------------------------------------------------===#
# FP matmul instructions (standard variants)
#===----------------------------------------------------------------------===#

# CHECK-ENCODING: encoding: [0x2b,0x24,0x04,0x08]
# CHECK-DISASM: th.mfmacc22.h
th.mfmacc22.h

# CHECK-ENCODING: encoding: [0x2b,0x28,0x04,0x08]
# CHECK-DISASM: th.mfmacc22.s.h
th.mfmacc22.s.h

# CHECK-ENCODING: encoding: [0x2b,0x28,0x84,0x08]
# CHECK-DISASM: th.mfmacc22.s.bf16
th.mfmacc22.s.bf16

# CHECK-ENCODING: encoding: [0x2b,0x28,0x08,0x08]
# CHECK-DISASM: th.mfmacc22.s
th.mfmacc22.s

#===----------------------------------------------------------------------===#
# INT matmul instructions
#===----------------------------------------------------------------------===#

# CHECK-ENCODING: encoding: [0x2b,0x28,0x80,0x19]
# CHECK-DISASM: th.mmacc22.w.b
th.mmacc22.w.b

# CHECK-ENCODING: encoding: [0x2b,0x28,0x00,0x18]
# CHECK-DISASM: th.mmaccu22.w.b
th.mmaccu22.w.b

# CHECK-ENCODING: encoding: [0x2b,0x28,0x80,0x18]
# CHECK-DISASM: th.mmaccus22.w.b
th.mmaccus22.w.b

# CHECK-ENCODING: encoding: [0x2b,0x28,0x00,0x19]
# CHECK-DISASM: th.mmaccsu22.w.b
th.mmaccsu22.w.b
