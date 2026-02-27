# Phase 3-4: Intrinsics and Clang Builtins

## LLVM IR Intrinsics (`IntrinsicsRISCVXTHeadMatrix.td`)

Defined with `TargetPrefix = "riscv"` and `int_riscv_th_` naming convention.

### Design approach
Matrix registers are not modeled as LLVM IR types. All matrix operations are modeled as side-effect-only intrinsics that operate on implicit matrix state. GPR operands (pointers, strides, scalars, immediates) are passed as explicit intrinsic arguments.

### Helper classes
- `THMatrix_NoArgs` - Pure matrix operations with no explicit operands (matmul, pack, conversions, element-wise MM, mmov.mm, mzero)
- `THMatrix_Load` - Matrix load: `(ptr base, int stride) -> void` with side effects
- `THMatrix_Store` - Matrix store: `(ptr base, int stride) -> void` with write-memory
- `THMatrix_Imm` - Operations with one 3-bit immediate (slide, broadcast, element-wise MVI)
- `THMatrix_ToGPR` - Matrix-to-GPR: reads matrix state, returns GPR value
- `THMatrix_FromGPR` - GPR-to-matrix: takes GPR value, writes matrix state

### Coverage summary (~230 intrinsics total)

| Category | Count | Details |
|----------|-------|---------|
| Configuration | 7 | mrelease + 6 msettile variants |
| Load | 28 | 7 types (A/B/C element/tile-stride + M whole-register) x 4 sizes |
| Store | 28 | 7 types x 4 sizes |
| FP matmul | 13 | Same-width, widening FP8/FP16/BF16/TF32/FP32 |
| INT matmul | 12 | INT4->INT32, INT8->INT32, INT16->INT64 (signed variants) |
| Partial/bypass matmul | 6 | Partial INT8->INT32, bypass INT |
| Misc (zero/move/pack/slide/broadcast) | 35 | mzero(4), mmov(9), mdup(4), mpack(3), slide(10), broadcast(5) |
| FP format conversions | 20 | FP8/FP16/BF16/FP32/FP64/TF32 inter-conversions |
| Float-int conversions | 12 | INT8<->FP16, INT16<->FP32 (signed/unsigned) |
| Fixed-point clip | 8 | MM and MVI variants (signed/unsigned, lower/upper) |
| Packed conversions | 8 | Pack/quad lower/upper (signed/unsigned) |
| Integer element-wise | 22 | 11 ops (add/sub/mul/mulh/max/min/shift) x 2 variants (MM/MVI) |
| FP element-wise | 30 | 5 ops (add/sub/mul/max/min) x 3 sizes (h/s/d) x 2 variants |

### Include wiring
Added to `IntrinsicsRISCV.td`:
```tablegen
include "llvm/IR/IntrinsicsRISCVXTHeadMatrix.td"
```

## Clang Builtins (`BuiltinsRISCVXTHeadMatrix.td`)

Defined with `__builtin_riscv_th_` prefix, gated by `"xtheadmatrix"` feature.

### Design approach
Same as intrinsics - matrix registers are not exposed as C types. All operations work on implicit matrix state. GPR operands are passed as explicit arguments.

### Prototype patterns
- `void()` - Pure matrix operations (matmul, conversions, element-wise MM)
- `void(void *, size_t)` - Load/store operations (base pointer + stride)
- `void(size_t)` - GPR-to-matrix operations (mmov.m.x, mdup.m.x)
- `void(unsigned int)` - Immediate operations (slide, broadcast, element-wise MVI)
- `size_t()` - Matrix-to-GPR operations (mmov.x.m)
- `size_t(size_t)` - Configuration operations (msettile*)

### Coverage
Full 1:1 coverage matching all LLVM IR intrinsics (~230 builtins total). All builtins are gated by the `"xtheadmatrix"` feature string and have `NoThrow` attribute.

### Include wiring
Added to `BuiltinsRISCV.td`:
```tablegen
include "clang/Basic/BuiltinsRISCVXTHeadMatrix.td"
```
