# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-xtheadmatrix %s 2>&1 \
# RUN:        | FileCheck %s

# Invalid register class tests
# GPR instead of matrix register for mzero
th.mzero a0
# CHECK: error:

# GPR instead of matrix register for mmov
th.mmov.mm a0, tr1
# CHECK: error:

# Matrix register instead of GPR for config
th.msettilem tr0
# CHECK: error:

# Matrix register instead of GPR for mmovb.x.m (now 3-operand)
th.mmovb.x.m tr0, tr1, tr2
# CHECK: error:

# Immediate out of range tests
# uimm3 out of range (>7)
th.mrslidedown tr0, tr1, 8
# CHECK: error:

th.mcslidedown.b tr0, tr1, 8
# CHECK: error:

th.madd.w.mv.i tr0, tr1, tr2, 8
# CHECK: error:

th.mfadd.s.mv.i tr0, tr1, tr2, 8
# CHECK: error:

th.mn4clipl.w.mv.i tr0, tr1, tr2, 8
# CHECK: error:

th.mrbca.mv.i tr0, tr1, 8
# CHECK: error:

# uimm10 out of range (>1023)
th.msettilemi 1024
# CHECK: error:

th.msettileki 2000
# CHECK: error:

th.msettileni -1
# CHECK: error:

# Negative immediate for uimm3
th.mrslidedown tr0, tr1, -1
# CHECK: error:

# Wrong number of operands
th.mfmacc.h tr0, tr1
# CHECK: error:

th.msettilem
# CHECK: error:

th.mzero
# CHECK: error:

th.mlae8 tr0, (a0)
# CHECK: error:

# Extra operands
th.mzero tr0, tr1
# CHECK: error:

th.mrelease a0
# CHECK: error:

# Invalid use of parentheses
th.mlae8 tr0, a0, a1
# CHECK: error:

# Load with matrix register as base
th.mlae8 tr0, (tr1), a1
# CHECK: error:

# Store with wrong register types
th.msae8 a0, (a1), a2
# CHECK: error:
